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
from sunwai.clients.rate_limit import api_mode as _api_mode, parse_retry_after, rate_gate
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
def _with_retries(fn, *, retries: int = None, base_delay: float = 1.0, deadline: float = None, call_timeout: float = 120.0, stage: str = None):
    """Call fn(timeout) with jittered backoff on 429/5xx/conn errors under a TOTAL deadline. Defaults: DUB_RETRIES=3 retries,
    DUB_RETRY_DEADLINE=150 s total (240 s in api mode). Each attempt's timeout is capped by the remaining budget. Returns result or raises.

    A 429/503 ``Retry-After`` is honoured (sleep max(backoff, Retry-After)); a 429 also pauses every process through the stage rate
    gate, and when the wait does not fit in the deadline this raises at once instead of hammering. 429s do not use up ``retries``;
    they have their own budget (DUB_429_RETRIES=5). In api mode a bare HTTP 500 is retried at most once (the hosted API answers 500
    for some client errors)."""
    retries = int(os.environ.get("DUB_RETRIES", "3")) if retries is None else retries
    if deadline is None:
        deadline = float(os.environ.get("DUB_RETRY_DEADLINE", "240" if _api_mode() else "150"))
    max429 = int(os.environ.get("DUB_429_RETRIES", "5"))
    t_end = time.time() + deadline
    last_exc = None
    attempt = n429 = 0
    while attempt <= retries:
        remaining = t_end - time.time()
        if remaining < 3:
            break
        ra = None
        try:
            resp = fn(max(3.0, min(call_timeout, remaining)))
            code = getattr(resp, "status_code", None)
            if code == 429 or (code is not None and 500 <= code < 600):
                last_exc = RuntimeError(f"HTTP {code}")
                if code in (429, 503):
                    ra = parse_retry_after((getattr(resp, "headers", None) or {}).get("Retry-After"))
                if code == 429:
                    n429 += 1
                    if ra is not None and stage:
                        g = rate_gate(stage)
                        if g is not None:
                            g.block(ra)
                elif code == 500 and _api_mode():
                    retries = min(retries, attempt + 1)
            else:
                resp.raise_for_status()
                return resp
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            last_exc = e
        except requests.exceptions.RequestException:
            raise  # 4xx other than 429: do not retry
        if n429 and getattr(last_exc, "args", None) == ("HTTP 429",):
            if n429 > max429:
                break
            delay = max(ra or 0.0, _random.uniform(0.5, 1.0) * base_delay * (2 ** min(n429 - 1, 4)))
            if time.time() + delay >= t_end:
                log.warning("Bodhan 429: Retry-After %.0fs does not fit the remaining %.0fs budget; giving up", delay, t_end - time.time())
                break
            log.warning("Bodhan 429 (Retry-After=%s); waiting %.1fs (429 retry %d/%d)", ra, delay, n429, max429)
            time.sleep(delay)
            continue
        if attempt >= retries:
            break
        delay = _random.uniform(0.5, 1.0) * base_delay * (2 ** attempt)
        if ra is not None:
            delay = max(delay, ra)
        attempt += 1
        if time.time() + delay >= t_end:
            break
        log.warning("Bodhan call failed (%s); retrying in %.1fs (attempt %d/%d)", last_exc, delay, attempt, retries)
        time.sleep(delay)
    raise RuntimeError(f"Bodhan API failed (retries<={retries}, deadline {deadline:.0f}s): {last_exc}") from last_exc


def bodhan_asr(session: requests.Session, key: str, wav_path: Path,
               language: Optional[str]) -> str:
    """Transcribe ``wav_path`` with the ASR service.

    Args:
        session: HTTP session to use.
        key: Bearer key for the ASR service.
        wav_path: WAV file to transcribe.
        language: Optional language hint.

    Returns:
        The stripped transcript (empty when the service returns none).
    """
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
    resp = _with_retries(do, stage="asr")
    js = resp.json()
    return js.get("text", "").strip()


def bodhan_mt(session: requests.Session, key: str, text: str, lang: str, *, deadline: float = None,
              call_timeout: float = 120.0, retries: int = None, t_end: float = None, budget=None) -> str:
    """Translate ``text`` into ``lang`` with the MT service.

    Args:
        session: HTTP session to use.
        key: Bearer key for the MT service.
        text: Source text.
        lang: Target language code.
        deadline: Overall time budget for retries, seconds.
        call_timeout: Timeout of a single call, seconds.
        retries: Maximum retries (service default when ``None``).
        t_end: Absolute ``time.time()`` after which no call starts.
        budget: Optional recovery budget that is credited for queue waits.

    Returns:
        The stripped translation.

    Raises:
        RuntimeError: Retries or the time budget were exhausted.
    """
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
        with inflight("mt", wait=w, rate_wait=(w if (budget is not None or t_end is not None) else None)):
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
    resp = _with_retries(do, retries=retries, deadline=deadline, call_timeout=call_timeout, stage="mt")
    js = resp.json()
    return js["choices"][0]["message"]["content"].strip()


def _tts_wav(b: bytes) -> bytes:
    """Return TTS audio as WAV PCM16 24 kHz mono (api mode; local replies pass through untouched).

    The hosted TTS labels its reply ``audio/mpeg`` although the bytes were RIFF/WAV. Trust the bytes: RIFF/WAVE is returned
    as is; anything else is decoded by ffmpeg; undecodable data (for example a JSON error body returned with 200) raises
    ``RuntimeError`` without the body text, so nothing sensitive leaks.
    """
    if not _api_mode():
        return b
    if len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WAVE":
        return b
    import subprocess
    import tempfile

    try:
        with tempfile.TemporaryDirectory() as td:  # seekable output: a pipe would leave a streaming (0x7fffffff frames) header
            src, dst = os.path.join(td, "in.bin"), os.path.join(td, "out.wav")
            with open(src, "wb") as f:
                f.write(b)
            p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", src, "-ac", "1", "-ar", "24000",
                                "-c:a", "pcm_s16le", dst], capture_output=True, timeout=60)
            if p.returncode == 0 and os.path.exists(dst):
                with open(dst, "rb") as f:
                    out = f.read()
            else:
                out = b""
    except Exception as e:
        raise RuntimeError(f"TTS reply is not WAV and cannot be converted ({type(e).__name__})") from None
    if len(out) < 100:
        raise RuntimeError("TTS reply is not decodable audio")
    return out


def bodhan_tts(session: requests.Session, key: str, text: str, lang: str,
               voice: str, style=None) -> bytes:
    """Synthesise ``text`` with the TTS service.

    Args:
        session: HTTP session to use.
        key: Bearer key for the TTS service.
        text: Text to speak.
        lang: Language code.
        voice: Voice name.
        style: Optional delivery style passed to the model.

    Returns:
        WAV (PCM16, 24 kHz, mono) bytes.
    """
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
    resp = _with_retries(do, retries=int(os.environ.get("DUB_TTS_RETRIES", "2")), stage="tts")
    return _tts_wav(resp.content)  # WAV PCM16 24 kHz mono bytes


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
        with inflight("tts", wait=max(0.0, min(timeout / 2, t_end - time.time())), rate_wait=max(0.0, t_end - time.time())):
            rem = t_end - time.time()
            if rem <= 0.05:
                raise requests.exceptions.Timeout("TTS split deadline reached while waiting for a slot")
            return _post_bounded(
                session, f"{_next_tts_base()}/v1/audio/speech", rem,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload, timeout=_budget_timeout(timeout, rem),
            )
    resp = _with_retries(do, retries=int(os.environ.get("DUB_TTS_RETRIES", "2")), deadline=max(1.0, t_end - time.time()),
                         call_timeout=max(1.0, min(120.0, t_end - time.time())), stage="tts")
    return _tts_wav(resp.content)


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
