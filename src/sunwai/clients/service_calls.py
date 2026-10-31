"""Calls to the Bodhan ASR, MT and TTS services with retries, deadlines and concurrency slots."""

from __future__ import annotations

import logging
import os
import random as _random
import time
from pathlib import Path
from typing import Dict, Optional

import requests

log = logging.getLogger(__name__)

from sunwai.clients import bodhan_endpoints as bodhan_cfg
from sunwai.clients.service_endpoints import (
    ASR_BASE,
    TTS_MODEL,
    _next_mt_base,
    _next_tts_base,
)
from sunwai.pipeline.support.concurrency import inflight


# ---------------------------------------------------------------------------
# Key loading (never logged)
# ---------------------------------------------------------------------------
def load_keys() -> Dict[str, str]:
    """Return the per-service Bodhan API keys (never logged); see ``bodhan_endpoints.load_keys``."""
    return bodhan_cfg.load_keys()


# ---------------------------------------------------------------------------
# HTTP helpers with retries
# ---------------------------------------------------------------------------
def _with_retries(fn, *, retries: int = None, base_delay: float = 1.0, deadline: float = None, call_timeout: float = 120.0):
    """Call fn(timeout) with jittered backoff on 429/5xx/conn errors under a TOTAL deadline. Defaults: DUB_RETRIES=3 retries, DUB_RETRY_DEADLINE=150 s total. Each attempt's timeout is
    capped by the remaining budget. Returns result or raises."""
    retries = int(os.environ.get("DUB_RETRIES", "3")) if retries is None else retries
    deadline = float(os.environ.get("DUB_RETRY_DEADLINE", "150")) if deadline is None else deadline
    t_end = time.time() + deadline
    last_exc = None
    for attempt in range(retries + 1):
        remaining = t_end - time.time()
        if remaining < 3:
            break
        try:
            resp = fn(max(3.0, min(call_timeout, remaining)))
            code = getattr(resp, "status_code", None)
            if code == 429 or (code is not None and 500 <= code < 600):
                last_exc = RuntimeError(f"HTTP {code}")
            else:
                resp.raise_for_status()
                return resp
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            last_exc = e
        except requests.exceptions.RequestException:
            raise  # 4xx other than 429: do not retry
        if attempt >= retries:
            break
        delay = _random.uniform(0.5, 1.0) * base_delay * (2 ** attempt)
        if time.time() + delay >= t_end:
            break
        log.warning("Bodhan call failed (%s); retrying in %.1fs (attempt %d/%d)", last_exc, delay, attempt + 1, retries)
        time.sleep(delay)
    raise RuntimeError(f"Bodhan API failed (retries<={retries}, deadline {deadline:.0f}s): {last_exc}") from last_exc


def bodhan_asr(session: requests.Session, key: str, wav_path: Path,
               language: Optional[str]) -> str:
    def do(timeout=120):
        with open(wav_path, "rb") as f:
            files = {"file": ("seg.wav", f, "audio/wav")}
            data = {"model": "indic-transcribe"}
            if language:
                data["language"] = language
            with inflight("asr", wait=timeout / 2):
                return session.post(
                    f"{ASR_BASE}/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {key}"},
                    files=files, data=data, timeout=timeout,
                )
    resp = _with_retries(do)
    js = resp.json()
    return js.get("text", "").strip()


def bodhan_mt(session: requests.Session, key: str, text: str, lang: str, *, deadline: float = None,
              call_timeout: float = 120.0, retries: int = None, t_end: float = None, budget=None) -> str:
    def do(timeout=120):
        payload = {
            "model": "indic-translate",
            "messages": [{"role": "user", "content": text}],
            "target_language": lang,
        }
        w = timeout / 2
        if budget is not None:                                         # queue wait is credited back to the budget
            w = max(0.0, min(w, budget.left()))
        elif t_end is not None:                                        # hard budget: wait and post never pass t_end
            w = max(0.0, min(w, t_end - time.time()))
        _tw = time.time()
        _wd = None
        with inflight("mt", wait=w):
            if budget is not None:
                budget.credit_wait(time.time() - _tw)
                _rem = budget.left()
                if _rem < 1.0:                                         # no call (and no 1 s floor) when the budget is gone
                    raise RuntimeError("MT script recovery: time budget exhausted")
                _wd = min(timeout, _rem)                               # body deadline = min(per-call timeout, remaining)
                timeout = _budget_timeout(timeout, _rem)
            elif t_end is not None:
                timeout = max(1.0, min(timeout, t_end - time.time()))
            _post = _stream_post if _wd is not None else (lambda ss, u, _d, **k: ss.post(u, **k))
            return _post(
                session, f"{_next_mt_base()}/v1/chat/completions", _wd,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload, timeout=timeout,
            )
    resp = _with_retries(do, retries=retries, deadline=deadline, call_timeout=call_timeout)
    js = resp.json()
    return js["choices"][0]["message"]["content"].strip()


def bodhan_tts(session: requests.Session, key: str, text: str, lang: str,
               voice: str, style=None) -> bytes:
    def do(timeout=120):
        import json as _json
        payload = {
            "model": TTS_MODEL,
            "input": text,
            "voice": voice,
            "instructions": _json.dumps({"lang": lang, **({"style": style} if style else {})}),
        }
        with inflight("tts", wait=timeout / 2):
            return session.post(
                f"{_next_tts_base()}/v1/audio/speech",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload, timeout=timeout,
            )
    resp = _with_retries(do, retries=int(os.environ.get("DUB_TTS_RETRIES", "2")))
    return resp.content  # WAV PCM16 24 kHz mono bytes


def _post_bounded(session, url: str, deadline_s: float, **kw):
    """budgeted POST for a NON-streaming TTS server. Time-to-headers is bounded by the
    (connect, read) tuple, which fits deadline_s; the body is then read in pieces with a monotonic-deadline check and a 1 s socket
    timeout, so the worst overrun past the deadline is ~1 s."""
    deadline = time.monotonic() + max(0.05, deadline_s)
    resp = session.post(url, stream=True, **kw)
    try:
        try:
            resp.raw._connection.sock.settimeout(1.0)
        except Exception:
            pass
        buf = []
        for b in resp.iter_content(chunk_size=8192):
            buf.append(b)
            if time.monotonic() > deadline:
                raise requests.exceptions.ReadTimeout("deadline: TTS body not complete in time")
        if time.monotonic() > deadline + 1.0:
            raise requests.exceptions.ReadTimeout("deadline exceeded")
        resp._content = b"".join(buf)
        resp._content_consumed = True
        return resp
    except BaseException:
        resp.close()
        raise


def _bodhan_tts_bounded(session: requests.Session, key: str, text: str, lang: str, voice: str, t_end: float) -> bytes:
    """TTS call for the per-sentence split ONLY (the legacy path keeps bodhan_tts above, unchanged). Hard bound: limiter wait,
    (connect, read) timeout tuple and a deadline-enforced body read all fit the time left before t_end; the
    remaining time is recomputed after the limiter wait; no retry starts after t_end."""
    def do(timeout=120):
        import json as _json
        payload = {"model": TTS_MODEL, "input": text, "voice": voice, "instructions": _json.dumps({"lang": lang})}
        with inflight("tts", wait=max(0.0, min(timeout / 2, t_end - time.time()))):
            rem = t_end - time.time()
            if rem <= 0.05:
                raise requests.exceptions.Timeout("TTS split deadline reached while waiting for a slot")
            return _post_bounded(
                session, f"{_next_tts_base()}/v1/audio/speech", rem,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload, timeout=_budget_timeout(timeout, rem),
            )
    resp = _with_retries(do, retries=int(os.environ.get("DUB_TTS_RETRIES", "2")), deadline=max(1.0, t_end - time.time()),
                         call_timeout=max(1.0, min(120.0, t_end - time.time())))
    return resp.content


def _stream_post(session, url: str, deadline_s: float, **kw):
    """budgeted POST with a DEADLINE-ENFORCED body read, no threads. The request is streamed with a small per-recv read
    timeout (<= 3 s) and the body is read in small pieces (read1: returns as soon as ANY bytes are available), checking the
    monotonic deadline after every piece; on expiry the response is closed and ReadTimeout raised. Worst-case overrun is one
    recv timeout (<= 3 s). The caller holds the inflight slot for the whole call (this runs inside `with inflight`)."""
    deadline = time.monotonic() + max(0.05, deadline_s)
    to = kw.pop("timeout")
    if isinstance(to, tuple):
        to = (to[0], max(0.2, min(3.0, to[1])))
    resp = session.post(url, stream=True, timeout=to, **kw)
    try:
        buf = []
        raw = resp.raw
        try:
            raw.decode_content = True                                 # gzip/br/deflate bodies must be decoded like requests does
        except Exception:
            pass
        rd = getattr(raw, "read1", None)
        if rd is not None:
            import socket as _sk
            import urllib3.exceptions as _u3
            try:
                while True:
                    if time.monotonic() > deadline:
                        raise requests.exceptions.ReadTimeout(f"deadline: response body not complete within {deadline_s:.1f}s")
                    b = rd(8192)
                    if not b:
                        break
                    buf.append(b)
            except requests.exceptions.RequestException:
                raise
            except (_u3.ReadTimeoutError, _sk.timeout, TimeoutError) as e:      # urllib3/socket errors -> requests errors, chain kept
                raise requests.exceptions.ReadTimeout(f"read timed out mid-body: {e}") from e
            except _u3.DecodeError as e:
                raise requests.exceptions.ContentDecodingError(str(e)) from e
            except (_u3.ProtocolError, _u3.HTTPError) as e:
                raise requests.exceptions.ChunkedEncodingError(str(e)) from e
        else:
            for b in resp.iter_content(chunk_size=256):
                buf.append(b)
                if time.monotonic() > deadline:
                    raise requests.exceptions.ReadTimeout(f"deadline: response body not complete within {deadline_s:.1f}s")
        if time.monotonic() > deadline + 3.0:
            raise requests.exceptions.ReadTimeout("deadline exceeded")
        resp._content = b"".join(buf)
        resp._content_consumed = True
        return resp
    except BaseException:
        resp.close()
        raise


def _budget_timeout(call_timeout: float, rem: float):
    """a scalar requests timeout applies to connect AND read separately. Budgeted POSTs use (connect, read) with
    connect + read <= min(call_timeout, remaining). Residual: read is an inactivity timeout, so a server trickling bytes can still
    overrun by up to one read period (accepted, documented)."""
    tot = max(0.0, min(call_timeout, rem))
    c = min(3.0, tot * 0.25)
    return (c, tot - c)
