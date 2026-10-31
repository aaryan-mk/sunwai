#!/usr/bin/env python3
"""
dub_pipeline.py — CPU-only MVP of a YouTube dubbing pipeline (no lip sync).

Stages: fetch -> audio -> vad -> slots -> speakers -> asr -> mt -> tts -> separate -> mix
All outputs are cached (skipped if present). Keys read from files, never printed.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import math
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import librosa
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
from sunwai.clients import bodhan_endpoints as bodhan_cfg
BASE_URL = bodhan_cfg.api_base()
def _envf(name, default, lo, hi):
    """finite, bounded float from the environment; blank/malformed/NaN/inf/out-of-range -> default."""
    try:
        v = float(os.environ.get(name, "").strip())
        if v == v and lo <= v <= hi:
            return v
    except Exception:
        pass
    return float(default)


from sunwai.clients.bodhan_endpoints import env_nonblank as _env_nonblank  # noqa: E402
from sunwai.paths import DATA_DIR  # noqa: E402

# Service endpoints come from one place (BODHAN_MODE api|local, see sunwai.clients.bodhan_endpoints).
ASR_BASE = bodhan_cfg.asr_base()
MT_BASES = bodhan_cfg.mt_bases()
MT_BASE = MT_BASES[0] if MT_BASES else ""
TTS_BASES = bodhan_cfg.tts_bases()
TTS_BASE = ",".join(TTS_BASES)
IT2_BASE = bodhan_cfg.it2_base()
IT2_TARGETS = set(os.environ.get("IT2_TARGETS", "mni,sd").split(","))
_TTS_RR = __import__("itertools").count()


def _next_tts_base():
    """Return the next TTS replica base URL (round robin)."""
    if not TTS_BASES:
        raise RuntimeError("no TTS endpoint configured")
    return TTS_BASES[next(_TTS_RR) % len(TTS_BASES)]


def _env_workers(name, default):
    """Positive-int env (blank/invalid -> default); an explicit 1 is honoured."""
    try:
        v = int(os.environ.get(name, "").strip())
        return v if v >= 1 else default
    except ValueError:
        return default


def _next_mt_base():
    """Return a random MT replica base URL (spreads load across serial replicas)."""
    if not MT_BASES:
        raise RuntimeError("no MT endpoint configured")
    return __import__("random").choice(MT_BASES)


LANGS = {"hi", "ta", "mr", "ur", "bn", "te"}
try:
    LANGS |= {t["code"] for t in json.load(open(DATA_DIR / "languages.json", encoding="utf-8"))["targets"]}
except Exception:
    pass

# Voice map
VOICE_MAP: Dict[Tuple[str, str], str] = {
    ("hi", "male"): "Amit",
    ("hi", "female"): "Kavya",
    ("ta", "male"): "Arun",
    ("ta", "female"): "Anitha",
    # others fall back to Amit/Kavya for male/female
}
DEFAULT_VOICE = {"male": "Amit", "female": "Kavya", "unknown": "Amit"}

log = logging.getLogger("dub_pipeline")

# ---------------------------------------------------------------------------
# hardening helpers
# ---------------------------------------------------------------------------
import hashlib
import random as _random
import threading as _threading

_TLS = _threading.local()


def _env_num(name, default, typ):
    """Positive finite number from env; blank/invalid/non-positive -> default."""
    try:
        v = typ(os.environ.get(name, "").strip())
        return v if v > 0 and v != float("inf") else default
    except ValueError:
        return default


class _SlotLimiter:
    """Cross-process counting semaphore. Every chunk is its own subprocess with its own thread pools, so
    per-process worker counts multiply by the number of concurrent chunks and requests pile up on a service without
    bound. A stage may hold at most `limit` in-flight calls across ALL processes on this host: `limit` lock files, each
    flock'ed while a call is in flight (kernel-released if the process dies, so a crash never leaks a slot).
    - ONE config per host: <slot_dir>/limits.json ({"tts": 40, ...}) overrides the env/default limit, so processes started
      with different environments still enforce the same cap (the first user writes the file from its own env/default).
    - FIFO fairness: a waiter takes a ticket (<stage>.q.<time_ns>.<pid>.<tid>, flock'ed while waiting) and only the
      smallest LIVE ticket polls for a slot, so an older request is never overtaken (dead tickets are reaped).
    - Waiting raises requests.exceptions.Timeout after `wait_s` so _with_retries accounts for the wait.
    - Fail-open: if the slot directory, lock file or flock is unusable, calls proceed unbounded (logged once).
    flock is host-local semantics on some filesystems; all chunk processes run on one host."""

    def __init__(self, stage, limit, slot_dir=None, wait_s=None):
        self.stage = stage
        self.dir = slot_dir or os.environ.get("DUB_SLOT_DIR", os.path.join(__import__("tempfile").gettempdir(), "sunwai_slots"))
        self._req_limit = max(1, int(limit))
        self.limit = self._req_limit
        self.wait_s = _env_num("DUB_SLOT_WAIT", 120.0, float) if wait_s is None else wait_s
        self._held = set()
        self._mu = _threading.Lock()
        self._broken = os.environ.get("DUB_SLOTS", "1") == "0"
        self._resolved = False

    def _fail_open(self, why):
        if not self._broken:
            log.warning("slot limiter for %s disabled (%s): calls are unbounded", self.stage, why)
        self._broken = True

    def _resolve_limit(self):
        """Host-wide limit from limits.json; create it (atomically, first writer wins) from this process's value."""
        if self._resolved:
            return
        path = os.path.join(self.dir, "limits.json")
        import fcntl
        lfd = None
        try:
            lfd = os.open(os.path.join(self.dir, "limits.lock"), os.O_CREAT | os.O_RDWR, 0o666)
            fcntl.flock(lfd, fcntl.LOCK_EX)           # merge under the lock: every stage gets ONE host-wide value
            try:
                cfg = json.load(open(path))
            except Exception:
                cfg = None
            if not isinstance(cfg, dict):
                cfg = {}
            if self.stage not in cfg:                 # first writer for THIS stage wins; never overwrite
                cfg[self.stage] = self._req_limit
                tmp = f"{path}.tmp{os.getpid()}_{_threading.get_ident()}"
                with open(tmp, "w") as _f:
                    json.dump(cfg, _f)
                os.replace(tmp, path)
        except OSError:
            cfg = cfg if isinstance(locals().get("cfg"), dict) else {}
        finally:
            if lfd is not None:
                os.close(lfd)
        try:
            self.limit = max(1, int(cfg.get(self.stage, self._req_limit)))
        except (TypeError, ValueError):
            self.limit = self._req_limit
        self._resolved = True

    def _head(self, mine):
        """True if no older LIVE ticket exists; dead tickets are removed."""
        import fcntl
        try:
            names = sorted(n for n in os.listdir(self.dir) if n.startswith(f"{self.stage}.q."))
        except OSError:
            return True
        for n in names:
            if n >= mine:
                break
            fd = None
            try:
                fd = os.open(os.path.join(self.dir, n), os.O_RDWR)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:                       # lockable = owner is gone
                    os.remove(os.path.join(self.dir, n))
                except OSError:
                    pass
            except BlockingIOError:
                return False               # an older waiter is alive
            except OSError:
                pass
            finally:
                if fd is not None:
                    os.close(fd)
        return True

    def _try_slots(self):
        import fcntl
        for i in range(self.limit):
            with self._mu:
                if i in self._held:
                    continue
                self._held.add(i)          # in-process claim first (flock may not conflict inside one process)
            fd = None
            try:
                fd = os.open(os.path.join(self.dir, f"{self.stage}.{i}.lock"), os.O_CREAT | os.O_RDWR, 0o666)
            except OSError as e:           # cannot even open the lock file (permissions, read-only fs): fail open
                with self._mu:
                    self._held.discard(i)
                self._fail_open(e)
                return "open"
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return (i, fd)
            except BlockingIOError:
                pass                       # another process holds slot i
            except OSError as e:
                import errno
                if e.errno not in (errno.EACCES, errno.EAGAIN):
                    os.close(fd)
                    with self._mu:
                        self._held.discard(i)
                    self._fail_open(e)
                    return "open"
            os.close(fd)
            with self._mu:
                self._held.discard(i)
        return None

    def acquire(self, wait=None):
        """-> (index, fd) or None when fail-open."""
        import fcntl
        if self._broken:
            return None
        try:
            os.makedirs(self.dir, exist_ok=True)
            self._resolve_limit()
        except OSError as e:
            self._fail_open(e)
            return None
        wait = self.wait_s if wait is None else min(self.wait_s, wait)
        t_end = time.time() + wait
        tname = f"{self.stage}.q.{time.time_ns():020d}.{os.getpid()}.{_threading.get_ident()}"
        tfd = None
        try:
            try:
                tfd = os.open(os.path.join(self.dir, tname), os.O_CREAT | os.O_RDWR, 0o666)
                fcntl.flock(tfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                self._fail_open(e)
                return None
            while True:
                if self._head(tname):
                    got = self._try_slots()
                    if got == "open":
                        return None
                    if got:
                        return got
                if time.time() >= t_end:
                    raise requests.exceptions.Timeout(f"no {self.stage} slot free after {wait:.0f}s ({self.limit} in flight)")
                time.sleep(_random.uniform(0.01, 0.04))
        finally:
            if tfd is not None:
                try:
                    os.remove(os.path.join(self.dir, tname))
                except OSError:
                    pass
                os.close(tfd)

    def release(self, tok):
        import fcntl
        if tok is None:
            return
        i, fd = tok
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
            with self._mu:
                self._held.discard(i)

    def __call__(self, wait=None):
        lim = self

        class _Ctx:
            def __enter__(self_):
                self_.tok = lim.acquire(wait)
                return self_

            def __exit__(self_, *a):
                lim.release(self_.tok)
        return _Ctx()


_LIMITERS = {}


def inflight(stage, wait=None):
    """`with inflight("tts"):` around ONE service call. Caps (env, across all chunk processes): ASR_MAX_INFLIGHT=8,
    MT_MAX_INFLIGHT=4 x replicas, TTS_MAX_INFLIGHT=4 x replicas; DUB_SLOTS=0 disables."""
    lim = _LIMITERS.get(stage)
    if lim is None:
        default = {"asr": 8, "mt": 4 * max(1, len(MT_BASES)), "tts": 4 * max(1, len(TTS_BASES))}[stage]
        lim = _LIMITERS.setdefault(stage, _SlotLimiter(stage, _env_num(f"{stage.upper()}_MAX_INFLIGHT", default, int)))
    return lim(wait)


def thread_session() -> requests.Session:
    """One requests.Session per thread (Session sharing across threads is not guaranteed thread-safe)."""
    ss = getattr(_TLS, "session", None)
    if ss is None:
        ss = requests.Session()
        ss.headers.update({"User-Agent": "dub-pipeline/1.0"})
        _TLS.session = ss
    return ss


def _fault(stage: str, seg_id) -> None:
    """Fault-injection hook for tests: DUB_FAULT="asr:3,mt:4,tts:5" makes that stage raise for that segment id."""
    spec = os.environ.get("DUB_FAULT", "")
    if spec and f"{stage}:{seg_id}" in [x.strip() for x in spec.split(",")]:
        raise RuntimeError(f"injected {stage} fault for seg {seg_id}")


def atomic_write_json(path, obj, **kw) -> None:
    """Write JSON to a temp file in the same dir, re-parse it, then os.replace (readers never see a partial file)."""
    path = str(path)
    tmp = f"{path}.tmp{os.getpid()}_{_threading.get_ident()}"
    kw.setdefault("ensure_ascii", False)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, **kw)
    with open(tmp, encoding="utf-8") as f:
        json.load(f)
    os.replace(tmp, path)


def media_ok(path, need_video=False, need_audio=True, min_dur=0.05) -> bool:
    """ffprobe validation: file exists, parses, has the required streams and duration >= min_dur."""
    import subprocess
    try:
        if not os.path.exists(path) or os.path.getsize(path) < 64:
            return False
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type:format=duration",
                            "-of", "json", str(path)], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return False
        js = json.loads(r.stdout or "{}")
        kinds = {x.get("codec_type") for x in js.get("streams", [])}
        if need_video and "video" not in kinds:
            return False
        if need_audio and "audio" not in kinds:
            return False
        return float(js.get("format", {}).get("duration") or 0.0) >= min_dur
    except Exception:
        return False


def failed_speech_frac(segments) -> Tuple[float, float, float]:
    """(failed_seconds, total_speech_seconds, fraction) over segments flagged fallback."""
    tot = sum(max(0.0, s["end"] - s["start"]) for s in segments)
    bad = sum(max(0.0, s["end"] - s["start"]) for s in segments
              if s.get("fallback") and s.get("fallback_reason") in ("asr", "mt", "tts"))
    return bad, tot, (bad / tot if tot > 0 else 0.0)


def enforce_fallback_budget(segments, stage: str) -> None:
    """Fail the chunk when failed speech exceeds FALLBACK_MAX_FRAC (default 0.20) of the chunk's speech."""
    bad, tot, frac = failed_speech_frac(segments)
    if bad > 0:
        log.warning("FALLBACK after %s: %d seg(s) %.1fs of %.1fs speech (%.0f%%) keep the ORIGINAL audio: %s", stage,
                    sum(1 for x in segments if x.get("fallback_reason") in ("asr", "mt", "tts")), bad, tot, 100 * frac,
                    [(x["id"], x.get("fallback_reason")) for x in segments if x.get("fallback_reason") in ("asr", "mt", "tts")])
    if frac > float(os.environ.get("FALLBACK_MAX_FRAC", "0.20")):
        raise RuntimeError(f"chunk failed: {frac:.0%} of speech ({bad:.1f}s/{tot:.1f}s) failed at {stage} (limit "
                           f"{float(os.environ.get('FALLBACK_MAX_FRAC', '0.20')):.0%})")


def mark_fallback(seg, reason: str, err) -> None:
    """Segment keeps the original audio: no dubbed speech, original vocals NOT masked, never speaks untranslated source."""
    seg["fallback"] = True
    seg["fallback_reason"] = reason
    seg["error"] = seg.get("error") or f"{reason}: {err}"
    seg[f"{reason}_error"] = str(err)


TTS_MODEL = os.environ.get("TTS_MODEL", "indic-speak")


def _expr_style(seg):
    """a style reaches TTS/fingerprints ONLY when the expression feature is ON and the style was set by the expression module
    (style_src == "m2"). Any other segment 'style' key (pre-existing data) is ignored: OFF is inert."""
    if _expr_on(seg) and seg.get("style_src") == "m2":
        st = seg.get("style")
        return st if isinstance(st, str) and st.strip() else None
    return None


_EXPR_FLAG = None


def _expr_flag():
    """ON only if EXPRESSION_ENABLED is truthy (default OFF).
    Evaluated once per process (each chunk runs in its own subprocess); any filesystem error counts as OFF."""
    global _EXPR_FLAG
    if _EXPR_FLAG is None:
        v = (os.environ.get("EXPRESSION_ENABLED") or "").strip().lower()
        if v in ("0", "false", "no", "off"):
            _EXPR_FLAG = False
        elif v in ("1", "true", "yes", "on"):
            _EXPR_FLAG = True
        else:
            _EXPR_FLAG = False                      # default OFF
    return _EXPR_FLAG


def _expr_on(seg):
    """expression feature ON, segment carries an m2 expression, and it was made for the CURRENT display text (a later text edit voids it)."""
    return (seg.get("tts_by") == "m2" and bool(seg.get("tts_text")) and seg.get("tts_src") == seg.get("tgt_text")
            and _expr_flag())


def _expr_spoken(seg):
    """text sent to TTS: the m2 tts_text only while _expr_on(seg); display text (tgt_text, subtitles) is never changed."""
    return seg["tts_text"] if _expr_on(seg) else _num_spoken(seg)      # v3: number words only for clear quantities


def tts_fingerprint(text: str, voice: str, lang: str, style=None) -> str:
    model = TTS_MODEL
    parts = [text, voice, lang, model] + ([style] if style else [])   # no style => fingerprint unchanged
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()


def write_raw_tts(raw_path: Path, wav_bytes: bytes, fp: str) -> None:
    """raw wav first (atomic), fingerprint sidecar second: a crash in between leaves a raw with no fp = regenerated."""
    tmp = raw_path.with_name(f".{raw_path.stem}.tmp{os.getpid()}_{_threading.get_ident()}.wav")
    tmp.write_bytes(wav_bytes)
    try:
        os.remove(raw_path.with_name(raw_path.stem + ".fp"))   # invalidate the old fingerprint BEFORE the audio changes
    except OSError:
        pass
    os.replace(tmp, raw_path)
    (raw_path.with_name(raw_path.stem + ".fp")).write_text(fp)


def raw_tts_fresh(raw_path: Path, fp: str) -> bool:
    try:
        return raw_path.exists() and raw_path.with_name(raw_path.stem + ".fp").read_text().strip() == fp
    except Exception:
        return False


def mix_mask(spans, N: int, sr: int = 44100) -> np.ndarray:
    """spans = [(start_sample, end_sample)] of regions where the original vocals are replaced (bed + TTS).
    Returns the Hann-smoothed 0..1 mask. The caller passes the END of the TTS audio when it spills past the slot
    (borrowed following silence) so spilled speech is NOT attenuated."""
    from scipy.signal.windows import hann
    cf = int(0.08 * sr)
    mask = np.zeros(N, dtype=np.float32)
    for a, b in spans:
        mask[max(0, a):min(N, b)] = 1.0
    win = hann(2 * cf + 1, sym=False)
    kernel = np.concatenate([win[:cf], win[cf:]])
    return np.clip(np.convolve(mask, kernel, mode="same") / kernel.sum(), 0.0, 1.0)

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

def _source_lang(out_dir):
    for d in (out_dir, os.path.dirname(os.environ.get("SPEAKER_MAP", "")) or None):
        try:
            lab = json.load(open(os.path.join(str(d), "lid.json"))).get("label") or ""
            return lab.split(":")[0].strip() or "en"
        except Exception:
            continue
    return "en"


def routed_mt(session, key, text, lang, src):
    if lang == "en":
        from sunwai.clients import llm as llm_client
        return llm_client.generate_text("Translate the following text into natural spoken English. Output only the translation.\n" + text,
                                        "translate", timeout=60, retries=5)
    if lang in IT2_TARGETS:
        try:
            r = session.post(f"{IT2_BASE}/v1/chat/completions", timeout=120,
                             json={"messages": [{"role": "user", "content": text}], "target_language": lang, "source_language": src})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log.warning("IndicTrans2 failed (%s); falling back to local MT", e)
    return bodhan_mt(session, key, text, lang)


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

# ---------------------------------------------------------------------------
# Audio IO helpers
# ---------------------------------------------------------------------------
def run_ffmpeg(args: List[str], *, capture: bool = True, check: bool = True):
    import subprocess
    inputs = {str(args[i + 1]) for i, a in enumerate(args[:-1]) if a == "-i"}
    out = str(args[-1])
    if out in inputs:
        # ffmpeg cannot write over its own input: render to a temp file, then replace
        tmp = out + ".tmp" + Path(out).suffix
        args = list(args[:-1]) + [tmp]
    cmd = ["ffmpeg", "-hide_banner", "-y", "-nostdin", "-loglevel", "error"] + args
    try:
        proc = subprocess.run(cmd, capture_output=capture, check=check,
                              timeout=_env_num("PIPE_FFMPEG_TIMEOUT", 900.0, float))
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg timed out after {_env_num('PIPE_FFMPEG_TIMEOUT', 900.0, float):.0f}s: {' '.join(cmd[-4:])}")
    if out in inputs and proc.returncode == 0:
        os.replace(tmp, out)
    return proc

def load_wav_mono(path: Path, sr: int = 16000) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=True, dtype=np.float32)
    return y

def load_wav_stereo(path: Path, sr: int = 44100) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=False, dtype=np.float32)
    if y.ndim == 1:
        y = np.stack([y, y])
    return y.T  # T x 2

def save_wav_mono(path: Path, y: np.ndarray, sr: int):
    import soundfile as sf
    sf.write(str(path), y, sr)

def save_wav_stereo(path: Path, y: np.ndarray, sr: int):
    import soundfile as sf
    if y.ndim == 1:
        y = np.stack([y, y], axis=1)
    sf.write(str(path), y, sr)

def crop_to_seconds(path: Path, out_path: Path, start: float, dur: float,
                    sr: int = 16000) -> np.ndarray:
    """Crop wav file to [start, start+dur] seconds at given sample rate."""
    y = load_wav_mono(path, sr=sr)
    a = int(round(start * sr))
    b = int(round((start + dur) * sr))
    b = min(b, len(y))
    a = max(0, min(a, b))
    return y[a:b]

# ---------------------------------------------------------------------------
# Stage 1: fetch
# ---------------------------------------------------------------------------
def _own_from_ts(t0, tb, ticks, r_fps, avg_fps, start, dur):
    """(D): PURE ownership rule on exact rationals. t0 = format start_time (Fraction, may be negative/non-zero), tb = stream
    time_base (Fraction), ticks = DECODED-frame timestamps (ints, presentation order, from ffprobe -show_frames best_effort_timestamp),
    r_fps/avg_fps = reported rates (Fraction), start/dur = chunk interval (Fraction). A frame is owned iff start <= ticks*tb - t0 < start+dur
    (strict, no tolerance). Raises ValueError (caller takes the legacy cut) on duplicate / non-monotonic / missing timestamps, non-CFR
    spacing, or reported fps that disagree with the measured spacing by > 0.5%. Returns (n_owned, delta_seconds_float)."""
    if len(ticks) < 3 or any(t is None for t in ticks):
        raise ValueError("too few / missing decoded timestamps")
    if any(b <= a for a, b in zip(ticks, ticks[1:])):
        raise ValueError("duplicate or non-monotonic decoded timestamps")
    if not (dur > 0 and tb > 0):
        raise ValueError("bad interval / time_base")
    d = [b - a for a, b in zip(ticks, ticks[1:])]
    med = sorted(d)[len(d) // 2]
    if max(abs(x - med) for x in d) * 50 > med:             # more than 2 % spacing jitter = not CFR
        raise ValueError("decoded frame spacing is not constant (VFR)")
    meas = 1 / (med * tb)                                   # measured fps
    for name, f in (("r_frame_rate", r_fps), ("avg_frame_rate", avg_fps)):
        if f is None or f <= 0 or abs(f - meas) * 200 > meas:
            raise ValueError(f"{name} {f} disagrees with the measured frame rate {float(meas):.4f}")
    own = [t for t in ticks if start <= t * tb - t0 < start + dur]
    if not own:
        raise ValueError("no frame inside the chunk")
    return len(own), float(own[0] * tb - t0 - start)


def _owned_frames_by_ts(src, start: float, dur: float):
    """(D): probe `src` (decoded frame timestamps) and apply _own_from_ts. Raises for anything unusable -> legacy cut."""
    import subprocess as _sp, json as _js
    from fractions import Fraction as _F
    if not (math.isfinite(start) and math.isfinite(dur) and start >= 0 and dur > 0):
        raise ValueError("bad chunk interval")
    st, du = _F(repr(float(start))), _F(repr(float(dur)))
    h = _sp.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "format=start_time:stream=time_base,r_frame_rate,avg_frame_rate", "-of", "json", str(src)],
                capture_output=True, text=True, timeout=60)
    j = _js.loads(h.stdout)
    t0 = _F(j["format"]["start_time"])
    s0 = j["streams"][0]
    tb, r_fps, avg_fps = _F(s0["time_base"]), _F(s0["r_frame_rate"]), _F(s0["avg_frame_rate"])
    lo = float(start) + float(t0) - 1.0
    rng = (f"{lo:.6f}%" if lo > 0 else "%") + f"{float(start) + float(dur) + float(t0) + 2.0:.6f}"
    p = _sp.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", rng, "-show_frames",
                 "-show_entries", "frame=best_effort_timestamp", "-of", "json", str(src)], capture_output=True, text=True, timeout=180)
    if p.returncode != 0:
        raise RuntimeError("ffprobe -show_frames failed")
    ticks = [(f.get("best_effort_timestamp") if isinstance(f.get("best_effort_timestamp"), int) else None) for f in _js.loads(p.stdout)["frames"]]
    return _own_from_ts(t0, tb, ticks, r_fps, avg_fps, st, du)


def stage_fetch(args, out_dir: Path) -> Dict[str, Any]:
    meta_path = out_dir / "meta.json"
    src_mp4 = out_dir / "src.mp4"

    if meta_path.exists() and src_mp4.exists():
        log.info("Stage fetch: cached")
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)

    url = args.url
    is_url = url.startswith("http://") or url.startswith("https://")

    if is_url:
        import yt_dlp
        ydl_opts = {
            "format": "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
            "outtmpl": str(out_dir / "raw_%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "merge_output_format": "mp4",
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            raw_path = Path(ydl.prepare_filename(info))
            title = info.get("title", "")
            uploader = info.get("uploader", "")
            license_ = info.get("license", "") or info.get("license_info", "")
    else:
        raw_path = Path(url)
        if not raw_path.exists():
            raise FileNotFoundError(f"Local input not found: {raw_path}")
        shutil.copy(raw_path, out_dir / raw_path.name)
        raw_path = out_dir / raw_path.name
        title = raw_path.stem
        uploader = ""
        license_ = ""

    start = args.start
    dur = args.dur

    # ffmpeg cut + re-encode
    # (D): the old cfr cut gave ONE FRAME TOO MANY per chunk (input -t is anchored at the first selected frame, so it reaches
    # into the next chunk) and rounded every picture to a frame slot (up to one frame of timing error). New: the owned frames are
    # chosen from the SOURCE's decoded timestamps (frame owned iff start <= pts - start_time < start+dur), passthrough video sync,
    # exactly that many frames, each at its true time. Only provably-CFR sources take this path; the result is verified (frame count)
    # and anything else falls back to the legacy command. CHUNK_EXACT_OWN=0 forces the legacy cut. Audio is cut exactly as before.
    _cut = [
        "-ss", f"{start}",
        "-t", f"{dur}",
        "-i", str(raw_path),
        "-map", "0:v:0?", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "veryfast",
        "-c:a", "aac",
    ]
    _why = None
    if os.environ.get("CHUNK_EXACT_OWN", "1").strip().lower() in ("0", "false", "off", "no"):
        _why = "disabled by CHUNK_EXACT_OWN=0"
    else:
        try:
            _n, _delta = _owned_frames_by_ts(raw_path, float(start), float(dur))
            _vf = "trim=end_frame=%d,setpts=PTS-STARTPTS+%.6f/TB" % (_n, _delta)
            run_ffmpeg(_cut + ["-vsync", "0", "-vf", _vf, str(src_mp4)])
            import subprocess as _sp
            _c = _sp.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries", "stream=nb_read_packets",
                          "-of", "csv=p=0", str(src_mp4)], capture_output=True, text=True, timeout=120).stdout.strip()
            if int(_c) != _n:
                raise RuntimeError(f"cut has {_c} frames, expected {_n}")
        except Exception as _e:
            _why = f"{type(_e).__name__}: {_e}"
    if _why is not None:
        if "disabled" not in _why:
            log.warning("fetch: exact frame cut rejected (%s); legacy cfr cut for this chunk", _why)
        try:
            os.remove(src_mp4)
        except OSError:
            pass
        run_ffmpeg(_cut + [str(src_mp4)])

    meta = {
        "title": title,
        "uploader": uploader,
        "license": license_,
        "url": url,
        "start": start,
        "dur": dur,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log.info("Stage fetch: done")
    return meta

# ---------------------------------------------------------------------------
# Stage 2: audio extraction
# ---------------------------------------------------------------------------
def _wav_ok(path) -> bool:
    try:
        import soundfile as _sf
        i = _sf.info(str(path))
        return i.frames > 0 and i.samplerate > 0
    except Exception:
        return False


def stage_audio(out_dir: Path):
    w16 = out_dir / "src16k.wav"
    w44 = out_dir / "src44k.wav"
    src_mp4 = out_dir / "src.mp4"
    if w16.exists() and w44.exists() and _wav_ok(w16) and _wav_ok(w44):
        log.info("Stage audio: cached")
        return

    for dst, args_ in ((w16, ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"]),
                       (w44, ["-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_f32le"])):
        tmp = out_dir / f".tmp{os.getpid()}_{dst.name}"
        try:
            run_ffmpeg(["-i", str(src_mp4)] + args_ + [str(tmp)])
            if not _wav_ok(tmp):
                raise RuntimeError(f"audio extraction produced an invalid {dst.name}")
            os.replace(tmp, dst)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
    log.info("Stage audio: done")

# ---------------------------------------------------------------------------
# Stage 3: VAD (Silero)
# ---------------------------------------------------------------------------
def stage_vad(out_dir: Path) -> List[Dict[str, float]]:
    vad_path = out_dir / "vad.json"
    if vad_path.exists():
        log.info("Stage vad: cached")
        with open(vad_path, encoding="utf-8") as f:
            return json.load(f)

    from silero_vad import load_silero_vad, get_speech_timestamps
    model = load_silero_vad()
    w16 = out_dir / "src16k.wav"

    import soundfile as sf
    # resource ceiling: a chunk is <= ~75 s; refuse a runaway input before reading it into RAM
    try:
        _max_s = float(os.environ.get("PIPE_MAX_AUDIO_S", "1800"))
        if not (0 < _max_s < float("inf")):
            raise ValueError
    except ValueError:
        _max_s = 1800.0
    _info = sf.info(str(w16))
    if _info.duration > _max_s:
        raise RuntimeError(f"chunk audio is {_info.duration:.0f}s, over the {_max_s:.0f}s ceiling (PIPE_MAX_AUDIO_S); "
                           f"the planner should have cut it into chunks")
    y, sr = sf.read(str(w16), dtype="float32")
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    y = y.reshape(-1)

    # get_speech_timestamps expects a torch tensor in recent versions
    import torch
    wav_t = torch.from_numpy(y)
    timestamps = get_speech_timestamps(
        wav_t, model,
        threshold=0.5,
        min_silence_duration_ms=100,
        min_speech_duration_ms=100,
        sampling_rate=16000,
        speech_pad_ms=0,
    )
    intervals = []
    for t in timestamps:
        s = t["start"] / sr
        e = t["end"] / sr
        intervals.append({"start": float(s), "end": float(e)})

    atomic_write_json(vad_path, intervals, indent=2)
    log.info("Stage vad: %d intervals", len(intervals))
    return intervals

# ---------------------------------------------------------------------------
# Stage 4: Slots (merge VAD intervals, pad for ASR, label pauses)
# ---------------------------------------------------------------------------
_NEWGEN = {"on": False}      # set by prep_shared while it builds a new-job chunk generation


CROP_HALFGAP = os.environ.get("CROP_HALFGAP", "1").strip().lower() not in ("0", "false", "off", "no")   # kill switch


def stage_slots(out_dir: Path, vad: List[Dict[str, float]]) -> List[Dict[str, Any]]:
    slot_path = out_dir / "slots.json"
    if slot_path.exists():
        log.info("Stage slots: cached")
        with open(slot_path, encoding="utf-8") as f:
            return json.load(f)

    total_dur = args_dur if (args_dur := (globals().get("_args_dur"))) else None

    # Merge VAD intervals into slots
    if not vad:
        with open(slot_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        return []

    merged: List[Dict[str, Any]] = []
    for iv in vad:
        s, e = iv["start"], iv["end"]
        if not merged:
            merged.append({"start": s, "end": e, "pauses": []})
            continue
        last = merged[-1]
        gap = s - last["end"]
        gap_s = max(0.0, gap)
        # classify pause
        if 0.15 <= gap_s <= 0.35:
            ptype = "comma"
        elif 0.35 < gap_s < 0.7:
            ptype = "sentence"
        else:
            ptype = None  # gap <0.15 or >=0.7

        if gap_s < 0.7 and (e - last["start"]) <= 12.0:
            # join
            last["end"] = max(last["end"], e)
            if ptype:
                last["pauses"].append({"gap": gap_s, "type": ptype})
        else:
            # split
            if ptype:
                merged.append({"start": s, "end": e, "pauses": [{"gap": gap_s, "type": ptype}]})
            else:
                merged.append({"start": s, "end": e, "pauses": []})

    # Force-cut runs >25 s at lowest-energy 20 ms frame between 18 and 25 s
    final: List[Dict[str, Any]] = []
    for sl in merged:
        if sl["end"] - sl["start"] <= 25.0:
            final.append(sl)
        else:
            pieces = _force_cut(sl, out_dir)
            final.extend(pieces)

    # Add crop padding (±0.2 s clamped) + assign ids + attach pauses of internal gaps
    for i, sl in enumerate(final):
        sl["id"] = i
        # crop with ±0.2 s
        cs = max(0.0, sl["start"] - 0.2)
        ce = sl["end"] + 0.2
        if CROP_HALFGAP:
            # never pad past the midpoint of the gap to a neighbour (a +-0.2 s pad on touching slots put the
            # neighbour's boundary word in BOTH crops -> repeated in two dubs). Gap <= 0 -> no pad on that side.
            if i > 0:
                cs = max(cs, sl["start"] - max(0.0, (sl["start"] - final[i - 1]["end"]) / 2.0))
            if i < len(final) - 1:
                ce = min(ce, sl["end"] + max(0.0, (final[i + 1]["start"] - sl["end"]) / 2.0))
        # clamp by neighbours / edges: don't overlap neighbour crops
        if i > 0:
            prev_end = final[i - 1]["end"]
            cs = max(cs, min(cs, prev_end - 0.01) if False else cs)  # keep simple: no strict overlap constraint for crop
        if i < len(final) - 1:
            nxt_start = final[i + 1]["start"]
            ce = min(ce, nxt_start + 0.2)
        # keep crop at least a bit larger than slot
        ce = max(ce, sl["end"] + 0.01)
        cs = min(cs, sl["start"] - 0.01)
        cs = max(0.0, cs)
        ce = min(ce, (total_dur or ce))
        if CROP_HALFGAP:        # the +-0.01 minimum pad above must not undo the half-gap on touching slots
            if i > 0:
                cs = max(cs, sl["start"] - max(0.0, (sl["start"] - final[i - 1]["end"]) / 2.0))
            if i < len(final) - 1:
                ce = min(ce, sl["end"] + max(0.0, (final[i + 1]["start"] - sl["end"]) / 2.0))
        sl["crop_start"] = float(cs)
        sl["crop_end"] = float(ce)

    # Ensure sorted by id
    final.sort(key=lambda x: x["start"])
    for i, sl in enumerate(final):
        sl["id"] = i

    # the speaker split runs ONLY while prep_shared is creating a NEW job's chunk generation (_NEWGEN["on"]).
    # Legacy chunks/jobs (and PREP_SHARE=0) never reach this branch: this function is then byte-for-byte the old behaviour.
    if _NEWGEN["on"]:
        final = split_slots_by_speaker(out_dir, final, vad, force=True)
    with open(slot_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    log.info("Stage slots: %d slots", len(final))
    return final

# ---------------------------------------------------------------------------
# split slots at diarization speaker changes (so each slot has ONE speaker, one voice, one start time)
# ---------------------------------------------------------------------------
SPLIT_SPK = os.environ.get("SPLIT_SPK", "1") == "1"   # default ON
SPLIT_MIN_SPK_S = _envf("SPLIT_MIN_SPK_S", 0.6, 0.1, 10.0)     # a speaker needs >= this much speech in the slot to count
SPLIT_MIN_PIECE_S = _envf("SPLIT_MIN_PIECE_S", 0.5, 0.1, 10.0)


def split_slots_by_speaker(out_dir: Path, slots: List[Dict[str, Any]], vad: List[Dict[str, float]],
                           min_spk: float = None, min_piece: float = None, force: bool = False) -> List[Dict[str, Any]]:
    """Split slots at diarization speaker changes. Pure function: returns the new list, writes nothing. Called ONLY from
    stage_slots at first creation of a NEW job's canonical slots (schema 2). Old caches are never re-split/renumbered."""
    smap = os.environ.get("SPEAKER_MAP")
    min_spk = SPLIT_MIN_SPK_S if min_spk is None else min_spk
    min_piece = SPLIT_MIN_PIECE_S if min_piece is None else min_piece
    if force and not (smap and os.path.exists(smap)):
        raise RuntimeError("schema-2 slots must be re-split but SPEAKER_MAP is unavailable")
    if not (smap and os.path.exists(smap) and slots):
        return slots
    try:
        g = json.load(open(smap, encoding="utf-8"))
        off = float(os.environ.get("CHUNK_START", "0"))
        turns = sorted((t["start"] - off, t["end"] - off, int(t["speaker"])) for t in g.get("turns", []))
        ivs = [((v["start"], v["end"]) if isinstance(v, dict) else (v[0], v[1])) for v in vad]
        out: List[Dict[str, Any]] = []
        changed = False
        for sl in slots:
            s, e = sl["start"], sl["end"]
            # who speaks when, inside the slot (clip turns)
            runs = []
            for a, b, spk in turns:
                a, b = max(a, s), min(b, e)
                if b - a <= 0:
                    continue
                if runs and runs[-1][2] == spk:
                    runs[-1][1] = max(runs[-1][1], b)
                else:
                    runs.append([a, b, spk])
            tot: Dict[int, float] = {}
            for a, b, spk in runs:
                tot[spk] = tot.get(spk, 0.0) + (b - a)
            keep = {k for k, v in tot.items() if v >= min_spk}
            runs = [r for r in runs if r[2] in keep]
            merged = []
            for r in runs:                       # re-merge after dropping short intruders
                if merged and merged[-1][2] == r[2]:
                    merged[-1][1] = max(merged[-1][1], r[1])
                else:
                    merged.append(list(r))
            if len(merged) < 2:
                out.append(sl)
                continue
            # boundary between consecutive runs: middle of the silence (or overlap) between them, snapped to the widest
            # VAD gap within +-0.6 s when one exists
            bounds = []
            for r0, r1 in zip(merged, merged[1:]):
                b = (r0[1] + r1[0]) / 2.0
                best = None
                for (a1, b1), (a2, b2) in zip(ivs, ivs[1:]):
                    gap = a2 - b1
                    if gap <= 0.05 or not (s < b1 and a2 < e):
                        continue
                    mid = (a2 + b1) / 2.0
                    if abs(mid - b) <= 0.6 and (best is None or gap > best[0]):
                        best = (gap, mid)
                bounds.append(best[1] if best else b)
            edges = [s] + bounds + [e]
            pieces = []
            for k in range(len(edges) - 1):
                lo, hi = edges[k], edges[k + 1]
                inside = [(max(a, lo), min(b, hi)) for a, b in ivs if min(b, hi) - max(a, lo) > 0.02]
                if inside:
                    ps, pe = inside[0][0], inside[-1][1]
                else:
                    ps, pe = lo, hi
                pauses = []
                for (a1, b1), (a2, b2) in zip(inside, inside[1:]):
                    gp = a2 - b1
                    if 0.15 <= gp <= 0.35:
                        pauses.append({"gap": gp, "type": "comma"})
                    elif 0.35 < gp < 0.7:
                        pauses.append({"gap": gp, "type": "sentence"})
                pieces.append({"start": ps, "end": pe, "pauses": pauses})
            if any(p["end"] - p["start"] < min_piece for p in pieces):
                out.append(sl)                    # a sliver piece would not be dubbable: keep the slot whole
                continue
            for k, p in enumerate(pieces):
                p["crop_start"] = sl["crop_start"] if k == 0 else max(0.0, p["start"] - min(0.2, max(0.0, (p["start"] - pieces[k - 1]["end"]) / 2.0)))
                p["crop_end"] = sl["crop_end"] if k == len(pieces) - 1 else p["end"] + min(0.2, max(0.0, (pieces[k + 1]["start"] - p["end"]) / 2.0))
                p["spk_split"] = True
            log.info("Stage slots: split slot %.2f-%.2f at speaker changes -> %s", s, e,
                     [(round(p["start"], 2), round(p["end"], 2)) for p in pieces])
            out.extend(pieces)
            changed = True
        if not changed:
            return slots
        out.sort(key=lambda x: x["start"])
        for i, sl in enumerate(out):
            sl["id"] = i
        return out
    except Exception as ex:
        if force:
            raise
        log.warning("speaker split skipped: %s", ex)
        return slots

FORCE_CUT_MAX_S = 25.0


def _cut_points(y, s: float, e: float, max_len: float = FORCE_CUT_MAX_S, sr: int = 16000) -> List[float]:
    """Cut times (ascending, strictly inside (s, e)) so that EVERY piece is <= max_len. Each cut is the lowest-energy 20 ms
    frame in [s+0.72*max_len, s+max_len] of the remaining run (18..25 s for the default); a degenerate search window falls
    back to s+max_len-1. Iterative, so the right-hand remainder is cut again."""
    cuts: List[float] = []
    frame_len = int(0.02 * sr)
    cur = s
    while e - cur > max_len + 1e-9:
        lo, hi = cur + 0.72 * max_len, min(cur + max_len, e)
        a = int(lo * sr)
        bnd = min(int(hi * sr), len(y) - frame_len)
        best_pos, best_energy = None, float("inf")
        for p in range(a, bnd, frame_len):
            frame = y[p:p + frame_len]
            energy = float(np.sum(frame * frame))
            if energy < best_energy:
                best_energy, best_pos = energy, p / sr
        if best_pos is None or best_pos <= cur + 1.0 or best_pos >= e - 0.05:
            best_pos = min(cur + max_len - 1.0, e - 0.05)
        cuts.append(best_pos)
        cur = best_pos
    return cuts


def _force_cut(sl: Dict[str, Any], out_dir: Path) -> List[Dict[str, Any]]:
    """Split a slot longer than FORCE_CUT_MAX_S (25 s) into pieces that are ALL <= 25 s, at low-energy frames (recursive)."""
    s, e = sl["start"], sl["end"]
    if e - s <= FORCE_CUT_MAX_S:
        return [sl]
    y = load_wav_mono(out_dir / "src16k.wav", sr=16000)
    edges = [s] + _cut_points(y, s, e) + [e]
    return [{"start": a, "end": b, "pauses": []} for a, b in zip(edges, edges[1:])]

# ---------------------------------------------------------------------------
# Stage 5: Speakers (ECAPA + clustering + gender)
# ---------------------------------------------------------------------------
def stage_speakers(out_dir: Path, slots: List[Dict[str, Any]]) -> Dict[str, Any]:
    sp_path = out_dir / "speakers.json"
    if sp_path.exists() and slots:
        log.info("Stage speakers: cached")
        with open(sp_path, encoding="utf-8") as f:
            return json.load(f)

    smap = os.environ.get("SPEAKER_MAP")
    if smap and os.path.exists(smap):
        g = json.load(open(smap, encoding="utf-8"))
        off = float(os.environ.get("CHUNK_START", "0"))
        turns = [(t["start"] - off, t["end"] - off, int(t["speaker"])) for t in g.get("turns", [])]
        ginfo = {int(x["id"]): x for x in g.get("speakers", [])}
        for sl in slots:
            ov = {}
            for a, b, spk in turns:
                o = min(b, sl["end"]) - max(a, sl["start"])
                if o > 0:
                    ov[spk] = ov.get(spk, 0.0) + o
            if ov:
                sl["speaker"] = max(ov, key=ov.get)
            elif turns:
                c = (sl["start"] + sl["end"]) / 2
                sl["speaker"] = min(turns, key=lambda t: abs((t[0] + t[1]) / 2 - c))[2]
            else:
                sl["speaker"] = 0
        info = {}
        for spk in sorted({sl["speaker"] for sl in slots}):
            mine = [sl for sl in slots if sl["speaker"] == spk]
            gs = ginfo.get(spk, {})
            info[str(spk)] = {"gender": gs.get("gender", "unknown"), "mean_p_female": gs.get("p_female", 0.5),
                              "total_speech": sum(sl["end"] - sl["start"] for sl in mine), "n_slots": len(mine), "global": True}
            for sl in mine:
                sl["gender"] = info[str(spk)]["gender"]
        with open(sp_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        log.info("Stage speakers: %d global speakers mapped from %s", len(info), smap)
        return info

    w16 = out_dir / "src16k.wav"
    y = load_wav_mono(w16, sr=16000)
    sr = 16000

    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(out_dir / ".ecapa"),
    )

    embs: Dict[int, Optional[np.ndarray]] = {}
    for sl in slots:
        dur = sl["end"] - sl["start"]
        if dur >= 0.8:
            a = int(sl["start"] * sr)
            b = int(sl["end"] * sr)
            seg = y[a:b]
            import torch
            emb = ecapa.encode_batch(torch.from_numpy(seg.reshape(1, -1)))
            embs[sl["id"]] = emb.squeeze().cpu().numpy()
        else:
            embs[sl["id"]] = None

    # Cluster only slots with embeddings
    ids_with = [sl["id"] for sl in slots if embs[sl["id"]] is not None]
    if ids_with:
        X = np.stack([embs[i] for i in ids_with])
        from sklearn.cluster import AgglomerativeClustering
        if len(ids_with) < 2:                   # sklearn cannot cluster a single embedding -> one speaker
            labels = np.zeros(len(ids_with), dtype=int)
        else:
            clustering = AgglomerativeClustering(
                metric="cosine", linkage="average",
                distance_threshold=0.6, n_clusters=None,
            )
            labels = clustering.fit_predict(X)
        label_map = dict(zip(ids_with, labels))
        max_label = max(labels) if len(labels) else 0
    else:
        label_map = {}
        max_label = -1

    # Assign short slots nearest slot speaker
    for sl in slots:
        if sl["id"] in label_map:
            sl["speaker"] = int(label_map[sl["id"]])
        else:
            # nearest by slot center
            c = (sl["start"] + sl["end"]) / 2
            best = None
            best_d = float("inf")
            for sl2 in slots:
                if sl2["id"] in label_map:
                    d = abs((sl2["start"] + sl2["end"]) / 2 - c)
                    if d < best_d:
                        best_d = d
                        best = sl2
            sl["speaker"] = int(label_map[best["id"]]) if best else 0     # best["speaker"] may not be assigned yet

    # Gender per speaker
    n_speakers = max_label + 1 if max_label >= 0 else (max((sl["speaker"] for sl in slots)) + 1 if slots else 0)
    gender_model = None
    if n_speakers > 0:                          # zero-speech clip -> no gender model load
        from model import ECAPA_gender
        gender_model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
        gender_model.eval()

    speakers_info: Dict[int, Dict[str, Any]] = {}
    for spk in range(n_speakers):
        spk_slots = [sl for sl in slots if sl["speaker"] == spk]
        # evidence: each >=1s, up to 10 slots
        eval_slots = [sl for sl in spk_slots if (sl["end"] - sl["start"]) >= 1.0][:10]
        total_speech = sum((sl["end"] - sl["start"]) for sl in spk_slots)
        p_female_scores = []
        for sl in eval_slots:
            a = int(sl["start"] * sr)
            b = int(sl["end"] * sr)
            seg = y[a:b]
            import torch
            inp = torch.from_numpy(seg.reshape(1, -1))
            with torch.no_grad():
                logits = gender_model(inp)
                probs = torch.softmax(logits, dim=-1).squeeze()
                pf = float(probs[1])
            p_female_scores.append(pf)
        mean_pf = float(np.mean(p_female_scores)) if p_female_scores else 0.5
        n_slots_ge1 = len(eval_slots)

        gender = "unknown"
        if mean_pf >= 0.9 and total_speech >= 8 and n_slots_ge1 >= 2:
            gender = "female"
        elif mean_pf <= 0.1 and total_speech >= 8 and n_slots_ge1 >= 2:
            gender = "male"
        speakers_info[spk] = {
            "gender": gender,
            "mean_p_female": mean_pf,
            "total_speech": float(total_speech),
            "n_slots": len(spk_slots),
        }

    with open(sp_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in speakers_info.items()}, f, indent=2)

    # attach speaker + gender back to slots (we'll store per-slot in segments later)
    for sl in slots:
        spk = sl["speaker"]
        sl["gender"] = speakers_info.get(spk, {}).get("gender", "unknown")

    with open(out_dir / "slots.json", "w", encoding="utf-8") as f:
        json.dump(slots, f, indent=2)
    log.info("Stage speakers: %d speakers", n_speakers)
    return speakers_info

# ---------------------------------------------------------------------------
# Stage 6: ASR
# ---------------------------------------------------------------------------
ASR_LANGS = {"en", "hi", "ta", "te", "kn", "ml", "mr", "bn", "gu", "pa", "or", "ur", "as", "ne", "sa"}


def _read_lid_record(path):
    """A published LID record, or None if absent / empty / torn / malformed."""
    try:
        d = json.load(open(path, encoding="utf-8"))
        return d if isinstance(d, dict) and "label" in d else None
    except Exception:
        return None


def _lid_compute(w16: Path, slots: List[Dict[str, Any]], out_dir: Path) -> Optional[Dict[str, Any]]:
    """The VoxLingua107 ECAPA LID on up to 60 s of VAD speech. Record {label, prob, seconds, code} or None (< 3 s speech)."""
    import torch
    from speechbrain.inference.classifiers import EncoderClassifier as _LID
    y = load_wav_mono(w16, 16000)
    parts, total = [], 0.0
    for sl in slots:
        a, b = int(sl["start"] * 16000), int(sl["end"] * 16000)
        parts.append(y[a:b]); total += (b - a) / 16000
        if total >= 60:
            break
    if total < 3:
        return None
    lid = _LID.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir=str(out_dir / ".lid"), run_opts={"device": "cpu"})
    out = lid.classify_batch(torch.from_numpy(np.concatenate(parts)).unsqueeze(0))
    prob = float(out[1].exp().item())
    label = out[3][0]
    code = label.split(":")[0].strip()
    log.info("Source LID: %s p=%.3f (%.1f s)", label, prob, total)
    return {"label": label, "prob": prob, "seconds": total, "code": code if (prob >= 0.7 and code in ASR_LANGS) else None}


def _lid_is_leader(out_dir) -> bool:
    """True when this chunk has the lowest cNNNN index present next to it (chunk 0 normally)."""
    import re
    try:
        m = re.fullmatch(r"c(\d+)", Path(out_dir).name)
        if not m:
            return True
        mine = int(m.group(1))
        idx = [int(x.group(1)) for x in (re.fullmatch(r"c(\d+)", d.name) for d in Path(out_dir).resolve().parent.iterdir()) if x]
        return mine <= min(idx + [mine])
    except Exception:
        return True


LID_LOCK = os.environ.get("LID_LOCK", "1").strip().lower() not in ("0", "false", "off", "no")


def _lid_first_writer(job_cache: str, compute, leader: bool = True, wait_s: float = 15.0) -> Optional[Dict[str, Any]]:
    """ONE language ID per job without delaying chunk 0. Only the leader chunk (lowest index present) computes and
    publishes (flock, tmp + fsync + os.replace, first writer wins). Every other chunk waits at most `wait_s` for the published
    record, then computes locally WITHOUT publishing. LID_LOCK=0 restores the old unlocked behaviour (compute + overwrite)."""
    rec = _read_lid_record(job_cache)
    if rec is not None:
        return rec
    if not LID_LOCK:
        rec = compute()
        if rec is not None:
            with open(job_cache, "w") as f:
                json.dump(rec, f)
        return rec
    lf = None
    t0 = time.time()
    deadline = t0 + wait_s          # ONE shared deadline for the lock attempt and the record wait
    if leader:
        lf = open(job_cache + ".lock", "a+")
        while True:
            try:
                fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.time() >= deadline:
                    lf.close(); lf = None
                    break
                time.sleep(0.2)
    if lf is None:
        while time.time() < deadline:
            rec = _read_lid_record(job_cache)
            if rec is not None:
                return rec
            time.sleep(0.25)
        rec = _read_lid_record(job_cache)
        return rec if rec is not None else compute()        # local, NOT published
    try:
        rec = _read_lid_record(job_cache)
        if rec is not None:
            return rec
        rec = compute()
        if rec is not None:
            tmp = f"{job_cache}.tmp{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(rec, f); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, job_cache)
        return rec
    finally:
        try:
            fcntl.flock(lf, fcntl.LOCK_UN)
        finally:
            lf.close()


def detect_source_language(w16: Path, slots: List[Dict[str, Any]], out_dir: Path) -> Optional[str]:
    """Video-level spoken-language ID (VoxLingua107 ECAPA) on up to 60 s of VAD speech; None unless confident."""
    # one LID per job: chunks of the same video share JOB_DIR/lid.json (next to the speaker map), first writer wins
    smap = os.environ.get("SPEAKER_MAP")
    job_cache = os.path.join(os.path.dirname(smap), "lid.json") if smap else None
    if job_cache:
        cached = _read_lid_record(job_cache) is not None
        rec = _lid_first_writer(job_cache, lambda: _lid_compute(w16, slots, out_dir), leader=_lid_is_leader(out_dir))
        if rec is None:
            return None
        if cached:
            log.info("Source LID (job cache): %s p=%.3f", rec.get("label"), rec.get("prob", 0))
        # the chunk copy mirrors the job record (so _source_lang agrees across chunks)
        atomic_write_json(out_dir / "lid.json", {"label": rec["label"], "prob": rec.get("prob", 0), "seconds": rec.get("seconds", 0)})
        return rec.get("code")
    rec = _lid_compute(w16, slots, out_dir)
    if rec is None:
        return None
    atomic_write_json(out_dir / "lid.json", {"label": rec["label"], "prob": rec["prob"], "seconds": rec["seconds"]})
    return rec["code"]


def _reset_downstream(segments):
    """A cached segments.json may carry MT/TTS fallback flags from an earlier failed attempt: those stages are re-run,
    so their failure state must not poison the retry. ASR flags are kept (and healed by re-running ASR)."""
    for sg in segments:
        if sg.get("fallback_reason") in ("mt", "tts"):
            for k in ("fallback", "fallback_reason", "mt_error", "tts_error"):
                sg.pop(k, None)
            sg["error"] = None
    return segments


ASR_DEDUP = os.environ.get("ASR_DEDUP", "0").strip().lower() in ("1", "true", "on", "yes")            # OFF by default (no word alignment yet)
ASR_LONG_SPLIT = os.environ.get("ASR_LONG_SPLIT", "1").strip().lower() not in ("0", "false", "off", "no")
_WORD_RE = None


def _asr_words(text):
    """[(normalised word, start, end)] spans of a transcript (letters/marks/digits/apostrophes, NFC, casefold)."""
    global _WORD_RE
    import unicodedata
    if _WORD_RE is None:
        import regex as _rx
        _WORD_RE = _rx.compile(r"[\p{L}\p{M}\p{N}]+(?:['\u2019][\p{L}]+)?")
    t = unicodedata.normalize("NFC", text or "")
    return t, [(unicodedata.normalize("NFC", m.group(0)).casefold(), m.start(), m.end()) for m in _WORD_RE.finditer(t)]


def dedup_boundary_words(segments, pad_per_word_s: float = 0.15, kmax: int = 6):
    """when two ADJACENT segments' ASR crops overlap by `ov` seconds, a boundary word can be transcribed in both.
    If the last k words of segment i equal the first k words of segment i+1 AND the overlap is long enough to hold k words
    (ov >= k * pad_per_word_s: a repeat longer than the shared audio is genuine speech, e.g. a stutter or refrain), the
    copy is dropped from the head of segment i+1 (the earlier segment keeps it) and recorded in seg["asr_dedup"]. Never empties
    a segment; segments already de-duplicated are skipped (idempotent). Returns the number of words dropped."""
    segs = sorted(segments, key=lambda x: x["id"])
    dropped = 0
    for a, b in zip(segs, segs[1:]):
        if b.get("asr_dedup") or b.get("edited") or b.get("fallback") or a.get("fallback"):
            continue
        ov = float(a.get("crop_end", 0)) - float(b.get("crop_start", 0))
        if ov <= 0:
            continue
        ta, wa = _asr_words(a.get("src_text"))
        tb, wb = _asr_words(b.get("src_text"))
        best = 0
        for k in range(1, min(kmax, len(wa), len(wb) - 1) + 1):
            if [w[0] for w in wa[-k:]] == [w[0] for w in wb[:k]] and ov >= k * pad_per_word_s:
                best = k
        if best:
            rest = tb[wb[best - 1][2]:].lstrip(" \t\n,;:.-\u2013\u2014")
            b["asr_dedup"] = {"dropped": tb[:wb[best - 1][2]], "words": best, "overlap_s": round(ov, 3), "prev": a["id"]}
            b["src_text"] = rest
            dropped += best
            log.info("ASR de-dup: seg %s lost %d duplicated boundary word(s) %r (crop overlap %.2fs with seg %s)", b["id"], best, b["asr_dedup"]["dropped"], ov, a["id"])
    return dropped


def stage_asr(args, out_dir: Path, slots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seg_path = out_dir / "segments.json"
    prior = None
    todo = slots
    if seg_path.exists():
        with open(seg_path, encoding="utf-8") as f:
            prior = _reset_downstream(json.load(f))
        failed_ids = {sg["id"] for sg in prior if sg.get("fallback_reason") == "asr"}
        if not failed_ids:
            log.info("Stage asr: cached")
            return prior
        log.info("Stage asr: cached, retrying %d slot(s) whose ASR failed earlier: %s", len(failed_ids), sorted(failed_ids))
        todo = [sl for sl in slots if sl["id"] in failed_ids]

    keys = load_keys()
    asr_key = keys["asr"]
    w16 = out_dir / "src16k.wav"
    crop_dir = out_dir / "crops"
    crop_dir.mkdir(exist_ok=True)

    MAX_ASR_SEC = 28.5
    try:
        src_lang = detect_source_language(w16, slots, out_dir)
    except Exception as e:
        log.warning("LID failed (%s); ASR auto-detects", e)
        src_lang = None

    def transcribe_slot(sl):
        cid = sl["id"]
        sl.pop("asr_error", None)
        try:
            crop_path = crop_dir / f"crop_{cid}.wav"
            cs = sl["crop_start"]
            ce = sl["crop_end"]
            if ce - cs > MAX_ASR_SEC and ASR_LONG_SPLIT:
                # a (legacy cached) slot longer than the ASR cap used to lose its tail; transcribe it in
                # consecutive pieces of <= 25 s cut at low-energy frames and join them
                yy = load_wav_mono(w16, 16000)
                edges = [cs] + _cut_points(yy, cs, ce) + [ce]
                texts = []
                for k, (pa, pb) in enumerate(zip(edges, edges[1:])):
                    pp = crop_dir / f"crop_{cid}_p{k}.wav"
                    save_wav_mono(pp, crop_to_seconds(w16, pp, pa, pb - pa, sr=16000), 16000)
                    _fault("asr", cid)
                    texts.append(bodhan_asr(thread_session(), asr_key, pp, src_lang))
                return cid, " ".join(t for t in texts if t).strip(), sl
            # ensure crop <= 28.5 s
            ce = min(ce, cs + MAX_ASR_SEC)
            y = crop_to_seconds(w16, crop_path, cs, ce - cs, sr=16000)
            save_wav_mono(crop_path, y, 16000)
            _fault("asr", cid)
            text = bodhan_asr(thread_session(), asr_key, crop_path, src_lang)
        except Exception as e:
            log.error("ASR error slot %d: %s", cid, e)
            text = ""
            sl["asr_error"] = str(e)
        return cid, text, sl

    # ONE ASR service: its own small worker count (was 2x TTS replicas = up to 20 concurrent calls)
    with ThreadPoolExecutor(max_workers=max(1, int(os.environ.get("ASR_WORKERS", "2")))) as ex:
        futures = [ex.submit(transcribe_slot, sl) for sl in todo]
        results = {}
        for fut in as_completed(futures):
            cid, text, sl = fut.result()
            results[cid] = (text, sl)

    def apply(sg, text, asr_error):
        sg["src_text"] = text
        for k in ("fallback", "fallback_reason", "asr_error"):
            sg.pop(k, None)
        sg["error"] = None
        if asr_error:
            sg["error"] = asr_error
            mark_fallback(sg, "asr", asr_error)
        elif not (text or "").strip():
            # VAD flagged speech but ASR heard none: nothing to dub, the original audio stays (labelled, not a failure)
            sg["fallback"], sg["fallback_reason"] = True, "no_speech"

    if prior is not None:
        segments = prior
        by_id = {sg["id"]: sg for sg in segments}
        for cid, (text, sl) in results.items():
            apply(by_id[cid], text, sl.get("asr_error"))
    else:
        segments = []
        for sl in slots:
            text, _ = results[sl["id"]]
            sg = {
                "id": sl["id"],
                "start": sl["start"],
                "end": sl["end"],
                "crop_start": sl["crop_start"],
                "crop_end": sl["crop_end"],
                "speaker": sl["speaker"],
                "gender": sl["gender"],
                "src_text": text,
                "pauses": sl.get("pauses", []),
                "tgt_text": "",
                "error": None,
                "edited": False,
                "history": [],
            }
            apply(sg, text, sl.get("asr_error"))
            segments.append(sg)
        segments.sort(key=lambda x: x["id"])

    if ASR_DEDUP:
        try:
            dedup_boundary_words(segments)
        except Exception as e:
            log.warning("ASR boundary de-dup skipped: %s", e)
    # fail the chunk BEFORE caching (a cached segments.json would freeze the failed ASR for every language), but keep a
    # diagnostic copy so the fallback markers/reasons are not lost
    try:
        enforce_fallback_budget(segments, "asr")
    except RuntimeError:
        try:
            atomic_write_json(out_dir / "segments.failed.json", segments, indent=2)
        except Exception:
            pass
        raise
    atomic_write_json(seg_path, segments, indent=2)
    log.info("Stage asr: %d segments", len(segments))
    return segments

# ---------------------------------------------------------------------------
# v3: number normalisation for TTS (spoken form only). Replaces v1/v2.
# ---------------------------------------------------------------------------
# review r1/r2 fixes: (1) a digit run is classified first; only clear QUANTITIES become words, identifiers (OTP/PIN/phone/codes,
# leading zeros, 5+ digits, separators, version/time/date shapes, years, 4-digit numbers without a unit) are left exactly as written;
# (2) no Gemini, no punctuation rewriting; (3) the spoken form never changes tgt_text (display, subtitles, fit guard input): it lives in
# seg["num_text"] with seg["num_src"] = the tgt_text it was made from (a refit/edit/resynth voids it); it is separate from the
# expression stage's tts_text; (4) the stage runs AFTER the fit and, when the spoken form would exceed the fit budget, the line is
# re-fitted with a smaller budget through the guarded rewrite, else it is NOT normalised; (5) bounded, locked cache.
# Kill switches: TTS_NORMALIZE=0 (everything), TTS_NORMALIZE_NUMBERS=0.
_TN_VER = "g091v3"
_TN_NUM_LANGS = ("hi", "mr", "en", "ta", "te", "bn")
_TN_PCT = {"hi": "प्रतिशत", "mr": "टक्के", "bn": "শতাংশ", "ta": "சதவீதம்", "te": "శాతం", "en": "percent"}
_TN_CACHE_MAX = 512
_TN_CACHE: "collections.OrderedDict" = __import__("collections").OrderedDict()
_TN_LOCK = __import__("threading").Lock()
_TN_INV: Dict[str, Dict[int, str]] = {}
_TN_ID_CUES = frozenset("""otp pin code id no no. number num phone mobile call dial account acct a/c room flight gate route version ver v model ticket order invoice zip pincode ref tel
ओटीपी पिन कोड नंबर नम्बर फोन मोबाइल खाता कमरा फ्लाइट ऑर्डर बिल क्रमांक आईडी
எண் கோட் ஓடிபி போன் மொபைல் அறை
নম্বর কোড ওটিপি ফোন মোবাইল""".split())
_TN_UNIT_AFTER = frozenset("""रुपये रुपए डॉलर लोग लोगों छात्र छात्रों मिनट घंटे घंटा सेकंड किलोमीटर किलो मीटर साल सालों वर्ष दिन दिनों बच्चे बच्चों आदमी किताबें किताब कॉपियाँ अंक वोट कॉपी ग्राम प्रतिशत बजे महीने महीनों हफ्ते सप्ताह देश देशों शहर शहरों गुना बार लीटर डिग्री सेंटीमीटर किलोग्राम पन्ने पृष्ठ सदस्य सदस्यों खिलाड़ी गोल मंज़िल मंजिल कमरे कुर्सियाँ सीटें
people students dollars rupees minutes hours seconds years days weeks months km kg meters metres miles percent users times words pages points members countries cities books children men women players goals floors rooms
ரூபாய் நிமிடம் நிமிட மணி நாட்கள் நாள் வருடம் ஆண்டு ஆண்டுகள் பேர் மக்கள் மாணவர்கள் கிலோமீட்டர் வயது வயதில் மாதங்கள் வாரங்கள் நாடுகள் புத்தகங்கள்
টাকা মিনিট ঘণ্টা দিন বছর জন লোক কিলোমিটার""".split())
_TN_CURRENCY_CODES = frozenset("usd inr eur gbp rs rs. aed".split())
_TN_SKIP_ONE = frozenset({"ta", "te"})              # 'one' before a noun is a different word (ஒரு / ఒక), table has the standalone form only

def _tn_flag(name: str, default: bool = True) -> bool:
    v = (os.environ.get(name, "") or "").strip().lower()
    if not v:
        return default
    return v not in ("0", "false", "off", "no", "n")




def _tn_inv(lang: str) -> Dict[int, str]:
    """value -> first number word of fit_stage._NUMV[lang] (insertion order); only int values."""
    inv = _TN_INV.get(lang)
    if inv is None:
        from sunwai.pipeline.fitting.fit_stage import _NUMV
        inv = {}
        for w, v in (_NUMV.get(lang) or {}).items():
            if isinstance(v, int):
                inv.setdefault(v, w)
        if lang == "hi" and "पाँच" in (_NUMV.get("hi") or {}):
            inv[5] = "पाँच"                      # the spelling people expect (table lists पांच first)
        _TN_INV[lang] = inv
    return inv


def _tn_int_words(n: int, lang: str):
    """Deterministic integer -> words, or None when the language table cannot express it exactly (caller then leaves the digits)."""
    inv = _tn_inv(lang)
    if n < 0 or not inv:
        return None
    if lang == "en":
        def lt1000(m):
            out = []
            if m >= 100:
                out += [inv[m // 100], inv[100]]; m %= 100
            if m >= 20:
                t10 = inv[m // 10 * 10]; m %= 10
                out.append(t10 + ("-" + inv[m] if m else ""))
            elif m or not out:
                out.append(inv[m])
            return out
        if n == 0:
            return inv[0]
        parts = []
        for scale, name in ((10**9, 10**9), (10**6, 10**6), (10**3, 10**3)):
            q, n = divmod(n, scale)
            if q:
                if q >= 1000:
                    return None
                parts += lt1000(q) + [inv[name]]
        if n:
            parts += lt1000(n)
        try:
            return " ".join(parts)
        except Exception:
            return None
    if lang in ("hi", "mr"):
        if n >= 10**9:
            return None
        parts = []
        try:
            for scale in (10**7, 10**5, 10**3):
                q, n = divmod(n, scale)
                if q:
                    parts += [inv[q], inv[scale]]          # q < 100 here, 0..99 are in the table
            if n >= 100:
                h, n = divmod(n, 100)
                if lang == "mr" and (h > 1 or n):
                    return None                              # mr 'she' hundreds forms are not in the table
                parts += [inv[100]] if lang == "mr" else [inv[h], inv[100]]
            if n or not parts:
                parts.append(inv[n])
        except KeyError:
            return None
        return " ".join(parts)
    return inv.get(n)                                        # ta / te / bn: only the exact table words


def _tn_words_to_int(tokens, lang):
    """Inverse (for tests / fallback verification): list of number-word tokens -> int, or None."""
    from sunwai.pipeline.fitting.fit_stage import _NUMV, _nfc
    tab = _NUMV.get(lang) or {}
    total = cur = 0
    tokens = [x for t in tokens for x in t.split("-") if x]
    for t in tokens:
        v = tab.get(_nfc(t).casefold()) if lang == "en" else tab.get(_nfc(t))
        if not isinstance(v, int):
            return None
        if v < 100:
            cur += v
        elif v == 100:
            cur = (cur or 1) * 100
        else:
            total += (cur or 1) * v; cur = 0
    return total + cur




def _tn_digits(tok: str) -> str:
    import unicodedata as _u
    return "".join(str(_u.decimal(c)) if c.isdecimal() else c for c in tok)




_TN_DOTTED_CUES = frozenset("no ref acct tel ph ver".split())


def _tn_cue_before(text: str, s: int) -> bool:
    """An identifier cue (OTP, PIN, code, phone, नंबर, No., ...) among the 3 words before the run, inside the same sentence/clause."""
    import regex as _rx
    if s > 0 and text[s - 1] == "#":
        return True
    pre = text[max(0, s - 240):s]
    cut = 0
    for m in _rx.finditer(r"[,;!?।\n\u2026]|\.{2,}|([\p{L}\p{M}]+)\.[\"'\u201d\u2019)\]]*(?=\s)", pre):
        if m.group(1) is None or m.group(1).casefold() not in _TN_DOTTED_CUES:
            cut = m.end()
    toks = _rx.findall(r"[\p{L}\p{M}/.]+", pre[cut:])[-3:]
    return any(t.casefold().strip(".") in _TN_ID_CUES for t in toks)


def _tn_next_word(text: str, e: int) -> str:
    import regex as _rx
    m = _rx.match(r"\s*([\p{L}\p{M}]+)", text[e:])
    return m.group(1).casefold() if m else ""


def _tn_quantity_words(text: str, m, lang: str):
    """Words for one digit run when it is a clear quantity, else None (the run stays exactly as written)."""
    import regex as _rx
    raw = m.group(1)
    if m.group(2) is not None:
        return None                                   # decimals: not handled
    d = _tn_digits(raw)
    s, e = m.start(1), m.end()
    prev = text[s - 1] if s > 0 else ""
    nxt = text[m.end(1)] if m.end(1) < len(text) else ""
    if (prev and prev in "-/:+@#_.%$\u20b9\u20ac\u00a3'\u2019\u066c\u066b") or (nxt and nxt in "-/:@#_'\u2019\u066c\u066b") or (nxt == "." and _rx.match(r"\.\p{Nd}", text[m.end(1):])):
        return None                                   # phone / time / date / version / code / currency-symbol shapes
    if _rx.search(r"\p{Nd}[ \u00a0\u202f]$", text[max(0, s - 2):s]) or _rx.match(r"[ \u00a0\u202f]\p{Nd}", text[m.end(1):]):
        return None                                   # space-separated digit groups ("1 500", "98765 43210")
    nw = _tn_next_word(text, m.end(1))
    if _tn_cue_before(text, s) or nw in _TN_ID_CUES or nw in _TN_CURRENCY_CODES:
        return None
    grouped = "," in d
    plain = d.replace(",", "")
    if grouped and not (_rx.fullmatch(r"\d{1,3}(,\d{3})+", d) or _rx.fullmatch(r"\d{1,2}(,\d{2})*,\d{3}", d)):
        return None
    if len(plain) > 1 and plain.startswith("0"):
        return None                                   # leading zero = identifier
    n = int(plain)
    unit = nw in _TN_UNIT_AFTER or bool(m.group(3) == "%")
    if not unit:
        return None                                   # conservative: a bare number could be a seat / bus / address / code; only numbers with a unit or % are quantities
    if not grouped and len(plain) >= 5:
        return None                                   # 5+ digits without grouping: phone / account / pin
    if lang in _TN_SKIP_ONE and n == 1:
        return None
    w = _tn_int_words(n, lang)
    if w is None:
        return None
    if m.group(3) == "%":
        pw = _TN_PCT.get(lang)
        if not pw:
            return None
        w = w + " " + pw
    return w


def _tn_numbers(text: str, lang: str) -> str:
    import regex as _rx
    if not _tn_inv(lang):
        return text
    pat = _rx.compile(r"(?<![\p{L}\p{M}\p{Nd}])(?<!\p{Nd}[,.])(\p{Nd}+(?:,\p{Nd}{2,3})*)(?:\.(\p{Nd}+))?(?![\p{L}\p{M}\p{Nd}])(?![,.]\p{Nd})(?:\s?(%)(?![\p{L}\p{M}\p{Nd}]))?")
    out, pos = [], 0
    for m in pat.finditer(text):
        w = _tn_quantity_words(text, m, lang)
        if w is None:
            continue
        out.append(text[pos:m.start()]); out.append(w); pos = m.end()
    out.append(text[pos:])
    return "".join(out)


def normalize_numbers(text: str, lang: str) -> str:
    """Spoken form of a display line: clear quantities written as words, everything else untouched. Never raises."""
    if not text or len(text) > 2000 or lang not in _TN_NUM_LANGS or not _tn_flag("TTS_NORMALIZE", True) or not _tn_flag("TTS_NORMALIZE_NUMBERS", True):
        return text
    ck = (_TN_VER, text, lang)
    with _TN_LOCK:
        hit = _TN_CACHE.get(ck)
        if hit is not None:
            _TN_CACHE.move_to_end(ck)
            return hit
    try:
        out = _tn_numbers(text, lang)
    except Exception as e:
        log.warning("normalize_numbers failed (%s): keeping the text", str(e)[:100])
        out = text
    with _TN_LOCK:
        _TN_CACHE[ck] = out
        while len(_TN_CACHE) > _TN_CACHE_MAX:
            _TN_CACHE.popitem(last=False)
    return out


def _num_spoken(seg: Dict[str, Any]) -> str:
    """tgt_text, or its number-normalised spoken form while ALL of these hold: it was made from the current tgt_text by this algorithm
    version and this process's target language, the switches are on and the segment is not edited / failed / fallback / expression."""
    t = seg.get("tgt_text") or ""
    if (seg.get("num_text") and seg.get("num_src") == t and seg.get("num_ver") == _TN_VER and seg.get("num_lang") == _NUM_STAGE_LANG
            and _NUM_STAGE_LANG is not None and not seg.get("edited") and not seg.get("error") and not seg.get("fallback")
            and not _expr_on(seg) and _tn_flag("TTS_NORMALIZE", True) and _tn_flag("TTS_NORMALIZE_NUMBERS", True)):
        return seg["num_text"]
    return t


_NUM_STAGE_LANG = None          # language of the last stage_numbers run in THIS process; a stored num_text is valid only for it


def stage_numbers(args, out_dir: Path, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """After the fit, deterministic, no external call, tgt_text (display, subtitles) is never modified. A line gets seg["num_text"]
    (+ num_src / num_ver / num_lang) only when (a) it contains a clear quantity, (b) fit.json holds a budget for exactly this fitted
    text and (c) the spoken form still fits that budget. Anything else is spoken exactly as today. If segments.json cannot be
    written, every spoken form is dropped (memory and disk must agree)."""
    global _NUM_STAGE_LANG
    from sunwai.pipeline.fitting.fit_stage import graphemes as _gr
    _NUM_STAGE_LANG = args.lang
    on = _tn_flag("TTS_NORMALIZE", True) and _tn_flag("TTS_NORMALIZE_NUMBERS", True)
    budgets = {}
    try:
        budgets = json.load(open(out_dir / "fit.json", encoding="utf-8"))
    except Exception:
        pass
    n = 0
    for s in segments:
        t = s.get("tgt_text") or ""
        for k in ("num_text", "num_src", "num_ver", "num_lang", "pre_numfit_text"):
            s.pop(k, None)
        if not on or not t.strip() or s.get("edited") or s.get("error") or s.get("fallback") or _expr_on(s):
            continue
        try:
            sp = normalize_numbers(t, args.lang)
            if sp == t:
                continue
            ent = budgets.get(str(s.get("id"))) or {}
            b = ent.get("budget")
            if not b or ent.get("fit_text") != t or _gr(sp) > b:
                continue                                       # no matching fit record, or the spoken form would exceed the fit budget
            s["num_text"], s["num_src"], s["num_ver"], s["num_lang"] = sp, t, _TN_VER, args.lang
            n += 1
        except Exception as e:
            log.warning("Stage numbers seg %s failed open (%s)", s.get("id"), str(e)[:100])
            for k in ("num_text", "num_src", "num_ver", "num_lang"):
                s.pop(k, None)
    try:
        atomic_write_json(out_dir / "segments.json", segments, indent=2)
    except Exception as e:
        log.warning("Stage numbers: segments.json write failed (%s); dropping spoken forms", e)
        for s in segments:
            for k in ("num_text", "num_src", "num_ver", "num_lang"):
                s.pop(k, None)
        n = 0
    log.info("Stage numbers: %d/%d lines have a spoken number form", n, len(segments))
    return segments


# ---------------------------------------------------------------------------
# Stage 7: MT
# ---------------------------------------------------------------------------
# the local indic-translate model copies the tail of a LONG Devanagari input through unchanged (Sameer c0003 seg 3,
# 27 words -> 44% Bengali). A wrong-script line used to get one identical retry and then count as a fallback. Recover first,
# under a hard per-segment time budget: (1) whole retry via the LOCAL model (skipped when routed_mt already used IT2),
# (2) <=MT_RECOVER_MAX_PIECES pieces of <=MT_PIECE_WORDS words translated with the LOCAL model, a bad piece repaired via IT2,
# (3) whole-segment IT2. EVERY accepted result must pass script_ok(strict=True); otherwise the segment is a fallback.
def _env_clamped(name: str, default, lo, hi, cast=float):
    try:
        v = cast((os.environ.get(name, "") or "").strip() or default)
    except (TypeError, ValueError):
        v = default
    return cast(max(lo, min(hi, v)))


MT_PIECE_WORDS = _env_clamped("MT_PIECE_WORDS", 12, 4, 40, int)
MT_RECOVER_BUDGET_S = _env_clamped("MT_RECOVER_BUDGET_S", 45.0, 5.0, 60.0)
MT_RECOVER_MAX_PIECES = _env_clamped("MT_RECOVER_MAX_PIECES", 6, 2, 12, int)
MT_RECOVER_CALL_S = _env_clamped("MT_RECOVER_CALL_S", 15.0, 1.0, 60.0)       # per-call POST timeout (limiter wait excluded)
MT_RECOVER_CAP_S = _env_clamped("MT_RECOVER_CAP_S", 120.0, 10.0, 180.0)      # hard WALL-CLOCK ceiling per segment (queue wait included)
MT_RECOVER_CHUNK_S = _env_clamped("MT_RECOVER_CHUNK_S", 180.0, 10.0, 600.0)   # total recovery work per chunk process (summed over segments)
_MT_SPENT = [0.0]          # chunk-wide pool already COMMITTED (spent + reserved by in-flight recoveries); guarded by _MT_SPENT_LOCK
_MT_SPENT_LOCK = _threading.Lock()
MT_RECOVER_EXTEND_S = _env_clamped("MT_RECOVER_EXTEND_S", 20.0, 0.0, 60.0)   # budget added per observed TIMEOUT (not per script error)


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


def _mt_pool_grant(want: float) -> float:
    """Reserve up to `want` s from the chunk-wide recovery pool (lock-protected); returns what was granted (may be 0)."""
    with _MT_SPENT_LOCK:
        g = max(0.0, min(want, MT_RECOVER_CHUNK_S - _MT_SPENT[0]))
        _MT_SPENT[0] += g
        return g


class _MtBudget:
    """Per-segment recovery budget with a HARD wall-clock cap (queue wait included) drawn from the shared per-chunk pool:
    it starts with min(base, pool) reserved; every genuine READ/CONNECT timeout and every credited queue wait asks the pool
    for more (granted only if the pool and the per-segment cap allow). finish() returns the unused part to the pool."""
    def __init__(self, base, cap, ext):
        self.t0 = time.time(); self.ext = ext; self.cap = max(1.0, cap); self.allow = 0.0
        self.allow += _mt_pool_grant(min(base, self.cap))

    def _grow(self, want: float):
        room = max(0.0, self.cap - self.allow)
        self.allow += _mt_pool_grant(min(want, room))

    def end(self) -> float:
        return self.t0 + min(self.allow, self.cap)

    def left(self) -> float:
        return self.end() - time.time()

    def credit_wait(self, s: float):
        self._grow(max(0.0, s))

    def timeout_seen(self):
        self._grow(self.ext)

    def finish(self):
        used = max(0.0, time.time() - self.t0)
        with _MT_SPENT_LOCK:
            _MT_SPENT[0] = max(0.0, _MT_SPENT[0] - max(0.0, self.allow - used))


def _is_timeout(e: BaseException) -> bool:
    """Only a genuine read/connect timeout (requests ReadTimeout/ConnectTimeout, socket.timeout), found by walking the exception
    chain (_with_retries re-raises as RuntimeError from the original). ConnectionError (refused/DNS), the limiter's queue-wait
    Timeout and 4xx/5xx are NOT timeouts, so they never extend the budget."""
    import socket as _socket
    seen = set()
    while e is not None and id(e) not in seen and len(seen) < 8:
        seen.add(id(e))
        if isinstance(e, (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout, _socket.timeout)):
            return True
        e = e.__cause__ or e.__context__
    return False


def _mt_pieces(text: str, n: int = None, max_pieces: int = None) -> List[str]:
    import re as _re
    n = n or MT_PIECE_WORDS

    def chunks(w, size):
        k = max(1, -(-len(w) // max(1, -(-len(w) // size))))        # even pieces of <= size words
        return [" ".join(w[i:i + k]) for i in range(0, len(w), k)]

    sents = [x.strip() for x in _re.split(r"(?<=[\u0964\u0965.?!;\u061f\u061b\u06d4])\s+", text or "") if x.strip()]
    out: List[str] = []
    for sent in sents:
        out.extend(chunks(sent.split(), n))
    if max_pieces and len(out) > max_pieces:
        words = (text or "").split()
        n2 = -(-len(words) // max_pieces)                              # raise the words-per-piece instead of skipping recovery
        out, cur = [], []
        for sent in sents:                                             # merge at sentence boundaries where possible
            w = sent.split()
            if len(w) > n2:
                if cur:
                    out.append(" ".join(cur)); cur = []
                out.extend(chunks(w, n2))
            elif len(cur) + len(w) > n2:
                out.append(" ".join(cur)); cur = list(w)
            else:
                cur += w
        if cur:
            out.append(" ".join(cur))
        if len(out) > max_pieces:                                      # fragmentation: plain even word chunks, exactly <= max
            out = [" ".join(words[i:i + n2]) for i in range(0, len(words), n2)]
    return out


def _it2_mt(session, text: str, lang: str, src: str, timeout: float = 20.0, t_end: float = None, budget=None) -> str:
    w = max(1.0, timeout / 2)
    if budget is not None:
        w = max(0.0, min(w, budget.left()))
    elif t_end is not None:
        w = max(0.0, min(w, t_end - time.time()))
    _tw = time.time()
    _wd = None
    with inflight("mt", wait=w):                                      # same concurrency limiter as MT
        if budget is not None:
            budget.credit_wait(time.time() - _tw)                     # the queue wait is not charged to the call
            _rem = budget.left()
            if _rem < 1.0:                                            # no call (and no 1 s floor) when the budget is gone
                raise RuntimeError("MT script recovery: time budget exhausted")
            _wd = min(timeout, _rem)                                  # body deadline = min(per-call timeout, remaining)
            timeout = _budget_timeout(timeout, _rem)
        elif t_end is not None:
            timeout = min(timeout, t_end - time.time())
        _post = _stream_post if _wd is not None else (lambda ss, u, _d, **k: ss.post(u, **k))
        r = _post(session, f"{IT2_BASE}/v1/chat/completions", _wd, timeout=timeout if budget is not None else max(1.0, timeout),
                  json={"messages": [{"role": "user", "content": text}], "target_language": lang, "source_language": src})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _mt_script_recover(session, key, text: str, lang: str, src: str, sid=None) -> str:
    from sunwai.pipeline.fitting.fit_stage import script_ok as _sok
    B = _MtBudget(MT_RECOVER_BUDGET_S, MT_RECOVER_CAP_S, MT_RECOVER_EXTEND_S)
    if B.allow < 2.0:                                                 # per-chunk recovery pool used up: fall back at once
        B.finish()
        log.warning("MT seg %s: chunk recovery allowance (%.0fs) exhausted; falling back", sid, MT_RECOVER_CHUNK_S)
        raise RuntimeError("MT output is in the wrong script for the target language")
    try:
        return _mt_script_recover_inner(session, key, text, lang, src, sid, B)
    finally:
        B.finish()


def _mt_script_recover_inner(session, key, text: str, lang: str, src: str, sid, B) -> str:
    from sunwai.pipeline.fitting.fit_stage import script_ok as _sok

    def left() -> float:
        r = B.left()
        if r < 2.0:
            raise RuntimeError("MT script recovery: time budget exhausted")
        return r

    def ok(x) -> bool:
        return bool((x or "").strip()) and _sok(x, lang, strict=True)

    def call(fn):
        """one retry on a TIMEOUT (a fresh full per-call timeout); every timeout extends the budget (bounded)."""
        for attempt in (0, 1):
            left()
            try:
                return fn(min(MT_RECOVER_CALL_S, max(1.0, B.left())))
            except Exception as e:
                if "budget" in str(e):
                    raise
                if _is_timeout(e):
                    B.timeout_seen()
                    if attempt == 0:
                        continue
                raise

    def local(txt):
        return call(lambda to: bodhan_mt(session, key, txt, lang, deadline=to * 1.3, call_timeout=to, retries=0, budget=B))

    def it2(txt):
        return call(lambda to: _it2_mt(session, txt, lang, src, timeout=to, budget=B))

    try:
        if lang not in IT2_TARGETS:                                   # for mni/sd the first call already was IT2
            try:
                t = local(text)
                if ok(t):
                    log.info("MT seg %s: recovered by whole local retry", sid)
                    return t
            except Exception as e:
                if "budget" in str(e):
                    raise
        pieces = _mt_pieces(text, max_pieces=MT_RECOVER_MAX_PIECES)
        if 1 < len(pieces) <= MT_RECOVER_MAX_PIECES:
            out = []
            for p in pieces:
                try:
                    q = local(p)
                except Exception as e:
                    if "budget" in str(e):
                        raise
                    q = ""
                if not ok(q):
                    try:
                        q = it2(p)
                    except Exception as e:
                        if "budget" in str(e):
                            raise
                        q = ""
                if not ok(q):
                    out = None
                    break
                out.append(q)
            if out:
                j = " ".join(out)
                if ok(j):
                    log.info("MT seg %s: recovered by %d-piece split (budget used %.0fs)", sid, len(pieces), time.time() - B.t0)
                    return j
        t = it2(text)
        if ok(t):
            log.info("MT seg %s: recovered by IndicTrans2 whole", sid)
            return t
    except Exception as e:
        log.warning("MT seg %s recovery stopped after %.0fs: %s", sid, time.time() - B.t0, str(e)[:120])
    raise RuntimeError("MT output is in the wrong script for the target language")


def stage_mt(args, out_dir: Path, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = load_keys()
    mt_key = keys["mt"]
    src = _source_lang(out_dir)
    log.info("Stage mt: source %s -> %s", src, args.lang)

    def translate(seg):
        if seg.get("edited"):
            return seg["id"], seg.get("tgt_text") or ""      # a user edit is never re-translated on a rerun
        if seg["error"]:
            return seg["id"], ""      # already a fallback (ASR failed): nothing to translate
        if not (seg.get("src_text") or "").strip():
            return seg["id"], ""      # ASR heard nothing: do not let the MT hallucinate speech from an empty line
        try:
            _fault("mt", seg["id"])
            t = routed_mt(thread_session(), mt_key, seg["src_text"], args.lang, src)
            from sunwai.pipeline.fitting.fit_stage import script_ok as _sok
            if (t or "").strip() and not _sok(t, args.lang):      # wrong-script MT output: retry, split, IT2 route; only then fail the segment
                log.warning("MT seg %s wrong script; recovering: %.60s", seg["id"], t)
                t = _mt_script_recover(thread_session(), mt_key, seg["src_text"], args.lang, src, seg["id"])
            if not (t or "").strip():
                raise RuntimeError("empty translation")
            from sunwai.pipeline.fitting.fit_stage import graphemes as _gr
            if (args.lang != src and " ".join(t.split()).casefold() == " ".join(seg["src_text"].split()).casefold()
                    and _gr(t) >= 12):
                raise RuntimeError("identity translation (untranslated source returned)")
        except Exception as e:
            # review: never speak the untranslated source; keep the original audio for this segment instead
            log.error("MT error seg %d: %s", seg["id"], e)
            t = ""
            mark_fallback(seg, "mt", e)
        return seg["id"], t

    with ThreadPoolExecutor(max_workers=max(2, int(os.environ.get("MT_WORKERS", str(len(MT_BASES)))))) as ex:
        futures = [ex.submit(translate, seg) for seg in segments]
        for fut in as_completed(futures):
            cid, t = fut.result()
            for seg in segments:
                if seg["id"] == cid:
                    seg["tgt_text"] = t
                    break

    atomic_write_json(out_dir / "segments.json", segments, indent=2)
    enforce_fallback_budget(segments, "mt")
    log.info("Stage mt: translated %d segments", len(segments))
    return segments

# ---------------------------------------------------------------------------
# Voice selection
# ---------------------------------------------------------------------------
_POOLS = None


def _voice_pools(lang: str):
    global _POOLS
    if _POOLS is None:
        try:
            cfg = json.load(open(DATA_DIR / "languages.json", encoding="utf-8"))
            _POOLS = {t["code"]: t.get("voices", {}) for t in cfg.get("targets", [])}
        except Exception:
            _POOLS = {}
    return _POOLS.get(lang, {})


# the voice plan is computed by the PLANNER (new jobs only) and embedded in speakers_global.json as "voice_plan".
# Workers only read it: validated once at preflight (a flagged job with a corrupt/incomplete/mismatched plan FAILS the chunk,
# there is no fallback to the legacy rule inside a planned job). No "voice_plan" key = legacy job = the exact old pick_voice.
_PLANNED_VOICES: Optional[Dict[str, str]] = None      # speaker id (str) -> voice for args.lang; None = legacy job


def voice_preflight(lang: str, out_dir: Optional[Path] = None) -> None:
    """Planned jobs are detected independently of SPEAKER_MAP (job-level rollout marker OR the map's own flag): a planned job
    whose map is missing/unreadable/unflagged FAILS. A job with neither marker nor flag is legacy."""
    global _PLANNED_VOICES
    _PLANNED_VOICES = None
    smap = os.environ.get("SPEAKER_MAP")
    jobdirs = []
    if smap: jobdirs.append(Path(os.path.dirname(smap)))
    if out_dir is not None: jobdirs.append(Path(out_dir).resolve().parent.parent)          # <job>/<lang>/cNNNN
    planned = False
    for jd in jobdirs:
        try:
            m = json.load(open(jd / "voice_rollout.json", encoding="utf-8"))
            planned = planned or (isinstance(m, dict) and m.get("voice_plan") == 1)
        except FileNotFoundError:
            pass
        except Exception as e:
            raise RuntimeError(f"voice_rollout.json unreadable ({jd}): {e}")
    g = None
    if smap and os.path.exists(smap):
        try:
            g = json.load(open(smap, encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"speaker map unreadable: {e}")
        planned = planned or ("voice_plan" in g)
    if not planned:
        return                                            # legacy job
    if g is None or "voice_plan" not in g:
        raise RuntimeError("planned job but SPEAKER_MAP is missing or carries no voice_plan (chunk fails, no legacy fallback)")
    from sunwai.planning import voices as VA
    try:
        plan = VA.validate_plan(g, lang)
    except VA.PlanError as e:
        raise RuntimeError(f"voice_plan invalid for this job (chunk fails, no legacy fallback): {e}")
    _PLANNED_VOICES = dict(plan["voices"][lang])
    log.info("voice plan: %s voices=%s degraded=%s", lang, _PLANNED_VOICES, plan.get("distinctness_degraded"))


def voice_require_speakers(segments) -> None:
    """planned job: every speaker id present in the chunk must be in the plan. Called OUTSIDE the per-segment handlers, so an
    unexpected id fails the whole chunk instead of degrading into a keep-original fallback."""
    if _PLANNED_VOICES is None:
        return
    missing = sorted({str(sg.get("speaker")) for sg in segments if str(sg.get("speaker")) not in _PLANNED_VOICES})
    if missing:
        raise RuntimeError(f"speaker id(s) {missing} are not in the voice plan (contract error; chunk fails)")


def pick_voice(lang: str, gender: str, speaker_idx: int) -> str:
    if _PLANNED_VOICES is not None:
        v = _PLANNED_VOICES.get(str(speaker_idx))
        if not v:
            raise RuntimeError(f"voice plan has no voice for speaker {speaker_idx} ({lang})")
        return v
    return _pick_voice_legacy(lang, gender, speaker_idx)


def _pick_voice_legacy(lang: str, gender: str, speaker_idx: int) -> str:
    pools = _voice_pools(lang)
    if pools:
        g = gender if gender in ("male", "female") else ("male" if speaker_idx % 2 == 0 else "female")
        pool = pools.get(g) or pools.get("male") or pools.get("female")
        if pool:
            return pool[(speaker_idx if gender in ("male", "female") else speaker_idx // 2) % len(pool)]
    key = (lang, gender)
    if key in VOICE_MAP:
        return VOICE_MAP[key]
    if gender in DEFAULT_VOICE:
        return DEFAULT_VOICE[gender]
    # unknown gender -> parity alternation for differentiation
    return "Amit" if (speaker_idx % 2 == 0) else "Kavya"

# ---------------------------------------------------------------------------
# per-sentence TTS placement, gentle slow-down / slack pauses, refit meaning-guard hook.
# Every behaviour has an env kill switch (read at call time, parsed robustly); all of it fails open to the old path.
# Edited segments (edited:true) never use any of it.
# ---------------------------------------------------------------------------
def _flag(name: str, default: str = "1") -> bool:
    v = os.environ.get(name)
    v = default if v is None or not v.strip() else v
    return v.strip().lower() in ("1", "true", "yes", "on")


_TTS_LAT: List[float] = []                    # durations (s) of this chunk's recent TTS calls (one process per chunk)
_TTS_LAT_LOCK = _threading.Lock()


def _record_lat(dt: float, wb=None) -> None:
    """latency bookkeeping: can never invalidate successful audio. Stores the real-time factor (call seconds per second of audio
    produced), so long lines are not mistaken for a loaded server."""
    try:
        aud = None
        try:
            import io as _io, wave as _wv
            _w = _wv.open(_io.BytesIO(wb))
            aud = _w.getnframes() / float(_w.getframerate())
        except Exception:
            aud = None
        if not aud or aud <= 0:
            return                                           # unknown output length: no sample (never guess)
        with _TTS_LAT_LOCK:
            _TTS_LAT.append(float(dt) / max(0.3, aud))
            del _TTS_LAT[:-8]
    except BaseException as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        try:
            log.warning("tts latency record skipped: %s", e)
        except Exception:
            pass


def _safe_cleanup(cid, tts_dir, made) -> None:
    """remove this attempt's piece files; never raises (logs only)."""
    try:
        files = [x for r_ in list(made or ()) for x in (r_, r_.with_name(r_.stem + ".fp"))]
        files += list(tts_dir.glob(f"{cid}_p*_trim.wav")) + list(tts_dir.glob(f"{cid}_p*_fit.wav"))
        for _p in files:
            try:
                os.remove(_p)
            except OSError:
                pass
    except Exception as e:
        try:
            log.warning("split cleanup skipped (seg %s): %s", cid, e)
        except Exception:
            pass


_SPLIT_SEM = None
_SPLIT_SEM_LOCK = _threading.Lock()


def _split_sem():
    global _SPLIT_SEM
    with _SPLIT_SEM_LOCK:
        if _SPLIT_SEM is None:
            _SPLIT_SEM = _threading.BoundedSemaphore(int(_envf("TTS_SENT_MAX_CONC", 3, 1, 8)))
        return _SPLIT_SEM


def _tts_slow_now() -> bool:
    """True when the rolling median real-time factor of the last 8 TTS calls exceeds TTS_SENT_SLOW_RTF (default 1.5 s of call per
    s of audio): splitting multiplies calls under load, so a loaded server gets legacy only."""
    with _TTS_LAT_LOCK:
        v = sorted(_TTS_LAT)
    return bool(v) and v[len(v) // 2] > _envf("TTS_SENT_SLOW_RTF", 1.5, 0.2, 100.0)


def _tts_slow_min() -> float:
    """lowest atempo for a slow-down (TTS shorter than its slot); 0.9 = natural-sounding limit. TTS_SLOW=0 disables."""
    return _envf("TTS_SLOW_MIN", 0.9, 0.9, 1.0) if _flag("TTS_SLOW") else 1.0


def _src_subspans(out_dir: Path, seg: Dict[str, Any], min_gap: float = None, max_pieces: int = None):
    """Source speech sub-spans of one slot, chunk time: VAD intervals clipped to the slot and joined unless separated by a
    silence >= min_gap. Returns None when the slot has no internal pause that long (or on any problem)."""
    min_gap = min_gap if min_gap is not None else _envf("TTS_SENT_MIN_GAP", 0.35, 0.2, 2.0)
    max_pieces = max_pieces if max_pieces is not None else int(_envf("TTS_SENT_MAX_PIECES", 4, 2, 8))
    try:
        iv = json.load(open(Path(out_dir) / "vad.json", encoding="utf-8"))
        iv = [((v["start"], v["end"]) if isinstance(v, dict) else (v[0], v[1])) for v in iv]
        s, e = float(seg["start"]), float(seg["end"])
        iv = sorted((max(a, s), min(b, e)) for a, b in iv if b > s + 0.01 and a < e - 0.01)
        iv = [x for x in iv if x[1] - x[0] > 0.02]
        if not iv:
            return None
        spans = [list(iv[0])]
        for a, b in iv[1:]:
            if a - spans[-1][1] >= min_gap:
                spans.append([a, b])
            else:
                spans[-1][1] = max(spans[-1][1], b)
        while len(spans) > max_pieces:                       # keep the longest pauses
            k = min(range(len(spans) - 1), key=lambda i: spans[i + 1][0] - spans[i][1])
            spans[k][1] = spans[k + 1][1]
            del spans[k + 1]
        return [(a, b) for a, b in spans] if len(spans) >= 2 else None
    except Exception:
        return None


_SPLIT_STRONG = ".?!;:।॥؟۔…"
_SPLIT_WEAK = ",،、—–"


def _g_count(txt: str) -> int:
    try:
        from sunwai.pipeline.fitting.fit_stage import graphemes
        return graphemes(txt)
    except Exception:
        return sum(1 for ch in txt if ch.isalnum())


def _split_target(text: str, spans, min_g: int = None, word_split: bool = None):
    """Split the translated line into len(spans) pieces so that piece k is spoken over source sub-span k. Boundaries are
    taken at sentence punctuation (then commas, then - only if word_split - any word boundary) nearest to the source's
    cumulative speech-time fraction. A boundary that cannot be placed within tolerance, or that leaves a piece with fewer
    than min_g graphemes, is dropped (neighbouring sub-spans merge). Returns [(piece_text, t0, t1)] or None (<2 pieces)."""
    import re as _re
    min_g = min_g if min_g is not None else int(_envf("TTS_SENT_MIN_G", 8, 2, 40))
    word_split = _flag("TTS_SENT_WORDSPLIT", "1") if word_split is None else word_split
    text = " ".join((text or "").split())
    K = len(spans)
    if K < 2 or not text:
        return None
    dur = [b - a for a, b in spans]
    tot = sum(dur) or 1e-6
    fr = [sum(dur[:i + 1]) / tot for i in range(K - 1)]            # cumulative speech-time fraction at each boundary
    toks = list(_re.finditer(r"\S+", text))
    G = max(1, _g_count(text))
    cands = []                                                      # (char_end, graphemes_before_frac, strength)
    for m in toks[:-1]:
        w = m.group().rstrip("\"'”’)]}")
        st = 2 if (w and w[-1] in _SPLIT_STRONG) else (1 if (w and w[-1] in _SPLIT_WEAK) else 0)
        if st == 0 and not word_split:
            continue
        cands.append((m.end(), _g_count(text[:m.end()]) / G, st))
    cuts, last = [], 0.0                                            # chosen (char_end, boundary index)
    for j, f in enumerate(fr):
        best = None
        for ce, cf, st in cands:
            if cf <= last + 0.03 or (cuts and ce <= cuts[-1][0]):
                continue
            tol = 0.12 if st == 0 else 0.22
            if abs(cf - f) > tol:
                continue
            cost = abs(cf - f) + (0.0, 0.02, 0.0)[st] + (0.06 if st == 0 else 0.0)
            if best is None or cost < best[0]:
                best = (cost, ce, cf)
        if best:
            cuts.append((best[1], j))
            last = best[2]
    if not cuts:
        return None
    # pieces of text; the sub-span list is merged across every dropped boundary
    kept = {j for _, j in cuts}
    groups, cur = [], [0]
    for i in range(1, K):
        if (i - 1) in kept:
            groups.append(cur); cur = [i]
        else:
            cur.append(i)
    groups.append(cur)
    ces = [0] + [ce for ce, _ in cuts] + [len(text)]
    pieces = [[text[ces[i]:ces[i + 1]].strip(), spans[g[0]][0], spans[g[-1]][1]] for i, g in enumerate(groups)]
    # drop pieces that are too small to synthesise well: merge into a neighbour
    i = 0
    while len(pieces) > 1 and i < len(pieces):
        if _g_count(pieces[i][0]) < min_g:
            j = i - 1 if i > 0 else i + 1
            a, b = (j, i) if j < i else (i, j)
            pieces[a] = [(pieces[a][0] + " " + pieces[b][0]).strip(), pieces[a][1], pieces[b][2]]
            del pieces[b]
            i = 0
        else:
            i += 1
    if len(pieces) < 2 or any(not p[0] for p in pieces):
        return None
    return [(p[0], p[1], p[2]) for p in pieces]


def _stretch_pauses(y: np.ndarray, sr: int, leftover: float, max_each: float = None, frac: float = None):
    """Spend part of `leftover` seconds of slack by lengthening the TTS's own internal pauses (>= 0.12 s of silence between
    speech), proportionally to their length, at most max_each per pause. Returns (new_y, added_s)."""
    max_each = max_each if max_each is not None else _envf("TTS_SLACK_MAX_PAUSE", 0.35, 0.05, 1.5)
    frac = frac if frac is not None else _envf("TTS_SLACK_FRAC", 0.7, 0.1, 1.0)
    db = _frame_db(y, sr)
    on, off, act = _speech_bounds(db, 30.0)
    if on is None:
        return y, 0.0
    runs, i = [], on
    while i < off:
        if not act[i]:
            j = i
            while j < off and not act[j]:
                j += 1
            if j - i >= 12:
                runs.append((i, j))
            i = j
        else:
            i += 1
    if not runs or leftover <= 0.03:
        return y, 0.0
    budget = leftover * frac
    lens = np.array([(b - a) * 0.01 for a, b in runs])
    add = np.minimum(max_each, budget * lens / lens.sum())
    # water-fill what the caps left over
    rest = budget - float(add.sum())
    for _ in range(3):
        room = max_each - add
        if rest <= 1e-3 or room.sum() <= 1e-6:
            break
        d = np.minimum(room, rest * room / room.sum()); add += d; rest -= float(d.sum())
    parts, prev = [], 0
    for (a, b), x in zip(runs, add):
        mid = int(((a + b) / 2) * 0.01 * sr)
        parts.append(y[prev:mid]); parts.append(np.zeros(int(x * sr), dtype=y.dtype)); prev = mid
    parts.append(y[prev:])
    return np.concatenate(parts), float(add.sum())


_MISSING = object()


class _SplitRestoreFailed(Exception):
    pass

_HOP = 0.01          # _frame_db default hop (s); the silence thresholds below are in these frames


def _whole_pause_drift(wav_path, spans, s0: float):
    """max |dub pause centre - source pause centre| (s) of the already-fitted whole-slot audio, pauses matched in order
    (ANY internal silence >= 0.12 s, matched in order to the N source gaps so that the largest offset is minimal). None when it cannot be told (unreadable); inf when the audio has
    fewer pauses than the source (cannot be aligned without splitting)."""
    try:
        import soundfile as _sf
        y, sr = _sf.read(str(wav_path), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        db = _frame_db(y, sr)
        on, off, act = _speech_bounds(db, 30.0)
        if on is None:
            return None
        runs, i = [], on
        while i < off:
            if not act[i]:
                j = i
                while j < off and not act[j]:
                    j += 1
                if j - i >= int(round(0.12 / _HOP)):
                    runs.append((i, j))
                i = j
            else:
                i += 1
        src = [((spans[k][1] + spans[k + 1][0]) / 2 - s0) for k in range(len(spans) - 1)]
        if len(runs) < len(src):
            return float("inf")
        cen = [((a + b) / 2) * _HOP for a, b in runs]
        # order-constrained assignment of the N source gaps to N of the dub's silences minimising the LARGEST offset (bottleneck DP):
        # extra breath/comma silences cannot displace the genuine pause, and the result is the best achievable max drift
        N, R = len(src), len(cen)
        INF = float("inf")
        best = [[INF] * (R + 1) for _ in range(N + 1)]
        for r in range(R + 1):
            best[0][r] = 0.0
        for n in range(1, N + 1):
            for r in range(n, R + 1):
                best[n][r] = min(best[n][r - 1], max(best[n - 1][r - 1], abs(cen[r - 1] - src[n - 1])))
        return best[N][R]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Stage 8: TTS + fit
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# runaway ladder: when every TTS retry still returns a runaway (indic-speak keeps generating, ~30.7 s for a 2-9 s slot), the
# pipeline used to KEEP the shortest runaway (garbage in the mix, overflow 15 s). Now: (1) sentence/clause split, each piece synthesised on
# its own (a runaway stays local to its piece); (2) if still too long, trim at slot*1.25 with a fade-out. Never keeps runaway audio.
# Kill switch PIPE_RUNAWAY_LADDER=0.
# ---------------------------------------------------------------------------
def runaway_ladder_on() -> bool:
    return os.environ.get("PIPE_RUNAWAY_LADDER", "1").strip().lower() not in ("0", "false", "no", "off", "n", "f")


def _runaway_gps() -> float:
    try:
        v = float(os.environ.get("PIPE_RUNAWAY_GPS", "8"))
        return v if 1.0 <= v <= 50.0 else 8.0
    except Exception:
        return 8.0


def is_runaway(d: float, slot: float, text: Optional[str] = None) -> bool:
    """Runaway TTS (indic-speak keeps generating). True when (a) the audio reaches the ~30.7 s server cap and is far beyond the slot, or
    (b) it is >4 s longer than the slot AND (text known) more than 2x what a SLOW reader (PIPE_RUNAWAY_GPS=8 chars/s) needs for that text
    -- a long text read at a normal pace is not a runaway and is left to the fit/refit path -- or (text unknown) more than 2.5x the slot."""
    if d >= 29.5 and d > slot * 1.5:
        return True
    if d <= slot + 4.0:
        return False
    if text:
        import re as _re
        g = len(_re.sub(r"\s+", "", text))
        return bool(g) and d > 2.0 * g / _runaway_gps()
    return d / max(slot, 1e-6) > 2.5


def runaway_split_pieces(text: str, n_max: int = 3, min_g: int = 12) -> List[str]:
    """Split a target text at clause/sentence boundaries into <= n_max balanced pieces (each >= min_g characters); [] if it cannot be split."""
    import re as _re
    parts = [p.strip() for p in _re.split(r"(?<=[.!?\u2026\u2014;:\u0964\u061f,])\s+|\s*\u2014\s*", text or "") if p and p.strip()]
    if len(parts) < 2:
        return []
    total = sum(len(p) for p in parts)
    target = total / min(n_max, len(parts))
    out, cur = [], ""
    for p in parts:
        cur = (cur + " " + p).strip() if cur else p
        if len(cur) >= max(min_g, target * 0.9) and len(out) < n_max - 1:
            out.append(cur); cur = ""
    if cur:
        if out and len(cur) < min_g:
            out[-1] = out[-1] + " " + cur
        else:
            out.append(cur)
    return out if len(out) >= 2 else []


def _wav_pcm(wb: bytes):
    import io as _io, wave as _wave
    w = _wave.open(_io.BytesIO(wb))
    return w.getframerate(), w.getnchannels(), w.getsampwidth(), w.readframes(w.getnframes())


def _wav_bytes(sr: int, ch: int, sw: int, pcm: bytes) -> bytes:
    import io as _io, wave as _wave
    b = _io.BytesIO()
    w = _wave.open(b, "wb"); w.setnchannels(ch); w.setsampwidth(sw); w.setframerate(sr); w.writeframes(pcm); w.close()
    return b.getvalue()


def wav_duration(wb: bytes) -> float:
    sr, ch, sw, pcm = _wav_pcm(wb)
    return len(pcm) / (ch * sw) / sr


def concat_wav_bytes(wbs: List[bytes], gap_s: float = 0.12) -> bytes:
    sr, ch, sw, first = _wav_pcm(wbs[0])
    gap = b"\x00" * (int(gap_s * sr) * ch * sw)
    parts = [first]
    for wb in wbs[1:]:
        sr2, ch2, sw2, pcm = _wav_pcm(wb)
        if (sr2, ch2, sw2) != (sr, ch, sw):
            raise ValueError("wav format mismatch")
        parts += [gap, pcm]
    return _wav_bytes(sr, ch, sw, b"".join(parts))


def trim_wav_bytes(wb: bytes, max_s: float, fade_s: float = 0.12) -> bytes:
    """Cut at max_s with a linear fade-out over the last fade_s (PCM16)."""
    sr, ch, sw, pcm = _wav_pcm(wb)
    if sw != 2:
        raise ValueError("only PCM16")
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    n = min(len(a), int(max_s * sr) * ch)
    a = a[:n]
    f = min(n, int(fade_s * sr) * ch)
    if f > 0:
        a[n - f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
    return _wav_bytes(sr, ch, sw, np.clip(a, -32768, 32767).astype(np.int16).tobytes())


def runaway_ladder(text: str, slot: float, best_wb: bytes, best_d: float, synth_piece, log=None) -> Tuple[bytes, float, str]:
    """-> (wav bytes, duration, how). synth_piece(piece_text, span_s) -> wav bytes (may raise). how: 'split' | 'trim' | 'none'."""
    say = (log.warning if log else (lambda *a, **k: None))
    pieces = runaway_split_pieces(text)
    if pieces:
        try:
            tot = sum(len(p) for p in pieces)
            wbs = [synth_piece(p, slot * len(p) / tot) for p in pieces]
            wb = concat_wav_bytes(wbs)
            d = wav_duration(wb)
            if not is_runaway(d, slot, text) and d < best_d * 0.8:
                return wb, d, "split"
            say("runaway ladder: split result still %.1fs for %.1fs slot", d, slot)
        except Exception as e:
            say("runaway ladder: split failed (%s)", e)
    cap = slot * 1.25 + 0.05
    try:
        wb = trim_wav_bytes(best_wb, cap)
        return wb, min(best_d, cap), "trim"
    except Exception as e:
        say("runaway ladder: trim failed (%s)", e)
        return best_wb, best_d, "none"


def stage_tts(args, out_dir: Path, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    voice_require_speakers(segments)
    tts_dir = out_dir / "tts"
    tts_dir.mkdir(exist_ok=True)
    keys = load_keys()
    session = None  # per-thread sessions via thread_session()
    tts_key = keys["tts"]

    def synth(seg):
        """Any exception while synthesising/fitting one segment degrades THAT segment to the original audio."""
        try:
            return _synth(seg)
        except Exception as e:
            log.error("TTS stage error seg %s: %s", seg.get("id"), e)
            mark_fallback(seg, "tts", e)
            seg.update(ratio=1.0, tempo=1.0, overflow=False, overflow_s=0.0)
            for _n in (f"{seg['id']}.wav", f"{seg['id']}_raw.wav"):
                try:
                    os.remove(tts_dir / _n)
                except OSError:
                    pass
            return seg["id"]

    def _tts_best(text, voice, span, t_end=None):
        """one TTS call (+ at most one runaway retry, same policy as the whole-slot path); returns wav bytes."""
        import io as _io, wave as _wave
        best, _t0 = None, time.time()
        for attempt in range(2):                                   # per piece: first call + at most ONE runaway retry
            if t_end is not None and time.time() > t_end:
                if best is None:
                    raise TimeoutError("split-slot deadline")
                break
            _t0c = time.time()
            wb = _bodhan_tts_bounded(thread_session(), tts_key, text, args.lang, voice, t_end)
            _record_lat(time.time() - _t0c, wb)
            w = _wave.open(_io.BytesIO(wb))
            d = w.getnframes() / w.getframerate()
            if best is None or d < best[1]:
                best = (wb, d)
            if d / max(span, 1e-6) <= 2.5 or time.time() - _t0 > float(os.environ.get("PIPE_RUNAWAY_MAX_S", "30")) or (t_end is not None and time.time() > t_end):
                break
        if best[1] < 0.05:
            raise RuntimeError("TTS returned empty audio")
        if runaway_ladder_on() and is_runaway(best[1], span, text):        # a runaway piece is trimmed, never placed
            log.error("TTS runaway [piece] %.1fs for %.1fs span -> trim", best[1], span)
            return trim_wav_bytes(best[0], span * 1.25 + 0.05)
        return best[0]

    def _synth_pieces(seg, voice, out_path, state) -> bool:
        """slot with an internal source pause >= TTS_SENT_MIN_GAP: synthesise each source-aligned piece on its own and
        place it at its source sub-span, writing ONE slot wav (the mix is unchanged). False -> caller uses the whole-slot path."""
        import soundfile as _sf
        cid = seg["id"]
        if _expr_on(seg):
            state["skip"] = "expr"
            return False                      # an expression line is synthesised whole-slot (tts_text + style), never split
        spans = _src_subspans(out_dir, seg)
        if not spans:
            state["skip"] = "no_pause"
            return False
        plan = _split_target(seg["tgt_text"], spans)
        if not plan:
            state["skip"] = "no_split_point"
            return False
        _dr = _whole_pause_drift(out_path, spans, float(seg["start"]))      # the legacy audio is already on disk: only upgrade when its pauses are really off
        if _dr is None:
            state["skip"] = "drift_unknown"                   # cannot tell whether the legacy pauses are off: keep legacy
            return False
        if _dr < _envf("TTS_SENT_MIN_DRIFT", 0.25, 0.0, 5.0):
            state["skip"] = "aligned"
            return False
        s0, slot_dur = float(seg["start"]), max(seg["end"] - seg["start"], 1e-6)
        room = slot_dur
        for s2 in segments:
            if s2["id"] == cid + 1:
                room = max(slot_dur, s2["start"] - seg["start"])
        sr = 24000
        slow_min, cursor, placed, info = _tts_slow_min(), 0.0, [], []
        t_end = time.time() + _envf("TTS_SENT_DEADLINE_S", 25.0, 5.0, 120.0)
        made = state.setdefault("made", set())          # files created during THIS attempt (pre-cached raws are never deleted)
        def _giveup(why):
            state["skip"] = why.split(" ")[0].replace(":", "")
            _safe_cleanup(cid, tts_dir, made)
            log.info("per-sentence TTS seg %s: %s; whole-slot path", cid, why)
            return False
        for k, (txt, a, b) in enumerate(plan):
            if time.time() > t_end:
                return _giveup("deadline")
            if _tts_slow_now():
                return _giveup("slow")
            raw = tts_dir / f"{cid}_p{k}_raw.wav"
            fp = tts_fingerprint(txt, voice, args.lang)
            span = max(b - a, 1e-3)
            if not raw_tts_fresh(raw, fp):
                _fault("tts", cid)
                try:
                    wbp = _tts_best(txt, voice, span, t_end)
                except TimeoutError:
                    return _giveup("deadline")
                if time.time() > t_end:                       # the call crossed the deadline: drop the split, one whole-slot attempt
                    return _giveup("deadline")
                write_raw_tts(raw, wbp, fp)
                made.add(raw)                                 # tracked only once this attempt has really written it
            trim = tts_dir / f"{cid}_p{k}_trim.wav"
            fit_src, d = _pretrim_wav(raw, trim)
            if k < len(plan) - 1:
                allowed = (plan[k + 1][1] - a) - 0.10
            else:
                allowed = room - (a - s0)
            ratio = d / span
            tempo = min(ratio, 1.25) if ratio > 1.0 else (max(ratio, slow_min) if ratio < 1.0 else 1.0)
            if allowed <= 0 or d / 1.25 > allowed + 0.02:     # cannot meet its sub-span even at 1.25x: never delay later pieces
                return _giveup("misfit")
            if d / tempo > allowed > 0:                       # still too long: speed up further into the pause that follows
                tempo = min(max(tempo, d / allowed), 1.25)
            tmpw = tts_dir / f"{cid}_p{k}_fit.wav"
            run_ffmpeg(["-i", str(fit_src), "-filter:a", f"atempo={tempo:.6f}", "-ac", "1", "-ar", str(sr), str(tmpw)])
            y, _sr = _sf.read(str(tmpw), dtype="float32")
            for _p in (tmpw, trim):
                try:
                    os.remove(_p)
                except OSError:
                    pass
            start = max(a - s0, (cursor + 0.10) if k > 0 else 0.0)
            cursor = start + len(y) / sr
            placed.append((start, y))
            info.append(dict(t0=round(a - s0, 3), t1=round(b - s0, 3), start=round(start, 3), dur=round(len(y) / sr, 3), tempo=round(tempo, 3), text=txt))
        if cursor > room + 0.05:                              # does not fit even with the pauses squeezed: let the whole-slot path (refit) decide
            return _giveup("room")
        out = np.zeros(int(cursor * sr) + 1, dtype=np.float32)
        fl = int(0.004 * sr)
        for start, y in placed:
            y = y.copy()
            if len(y) > 2 * fl:
                y[:fl] *= np.linspace(0, 1, fl, dtype=y.dtype); y[-fl:] *= np.linspace(1, 0, fl, dtype=y.dtype)
            i0 = int(start * sr)
            out[i0:i0 + len(y)] += y[:len(out) - i0]
        tmp = out_path.with_name(f".{out_path.stem}.tmp{os.getpid()}_{_threading.get_ident()}.wav")
        made.add(tmp)                                  # removed by the guarded cleanup if the write/replace fails
        _sf.write(str(tmp), out, sr, subtype="PCM_16")
        state["replaced"] = True                       # pre-armed BEFORE the replace: from here on only the backup can bring the legacy audio back
        os.replace(tmp, out_path)
        seg["sent_pieces"] = info
        seg.pop("slack_added", None)
        seg["ratio"] = cursor / slot_dur
        seg["tempo"] = max(p["tempo"] for p in info)
        seg["overflow"] = False
        seg["overflow_s"] = 0.0
        seg["voice"] = voice
        log.info("per-sentence TTS seg %s: %d pieces, end %.2fs / slot %.2fs", cid, len(info), cursor, slot_dur)
        return True

    def _runaway_guard(seg, voice, raw_path, where):
        """uniform runaway check on the raw TTS file that is about to be fitted (fresh, CACHED, or after a refit). A runaway is never
        placed: split -> trim (shared 60 s deadline); if neither works the exception degrades this segment to the original audio."""
        if not runaway_ladder_on():
            return False
        import io as _io, wave as _wave
        cid = seg["id"]
        slot = max(seg["end"] - seg["start"], 1e-6)
        wb = Path(raw_path).read_bytes()
        d = wav_duration(wb)
        text = _expr_spoken(seg)
        if not is_runaway(d, slot, text):
            return False
        log.warning("TTS runaway [%s] seg %d: %.1fs for %.1fs slot", where, cid, d, slot)
        _end = time.time() + float(os.environ.get("PIPE_RUNAWAY_LADDER_S", "60"))          # ONE deadline for the whole ladder
        if _expr_on(seg):
            _synth_piece = lambda _p, _span: (_ for _ in ()).throw(RuntimeError("expression line: trim only"))
        else:
            _synth_piece = lambda _p, _span: _tts_best(_p, voice, _span, _end)
        nb, nd, how = runaway_ladder(text, slot, wb, d, _synth_piece, log)
        if how == "none":
            for _p in (Path(raw_path), Path(raw_path).with_suffix(".fp")):      # never leave the runaway cached: the next run re-synthesises
                try:
                    os.remove(_p)
                except OSError:
                    pass
            raise RuntimeError(f"runaway TTS ({d:.1f}s for a {slot:.1f}s slot) could not be repaired")
        log.error("TTS runaway [%s] seg %d: %.1fs for %.1fs slot -> ladder '%s' (%.1fs)", where, cid, d, slot, how, nd)
        write_raw_tts(Path(raw_path), nb, tts_fingerprint(text, voice, args.lang, _expr_style(seg)))
        seg["runaway_fix"] = how
        return True

    def _synth(seg):
        cid = seg["id"]
        state = {"one_try": False}
        out_path = tts_dir / f"{cid}.wav"
        raw_path = tts_dir / f"{cid}_raw.wav"
        if not (seg.get("tgt_text") or "").strip():
            seg.pop("sent_pieces", None); seg.pop("slack_added", None); seg.pop("legacy_fit", None); seg.pop("sent_skip", None)
            seg.update(ratio=1.0, tempo=1.0, overflow=False, overflow_s=0.0)
            for _p in (out_path, raw_path):
                try:
                    os.remove(_p)      # never leave a stale wav that the mix would pick up
                except OSError:
                    pass
            return cid   # nothing to speak (empty source): no TTS, original audio is kept
        _voice0 = pick_voice(args.lang, seg["gender"], seg["speaker"])
        seg.pop("sent_pieces", None)
        seg.pop("sent_skip", None)
        seg.pop("legacy_fit", None)
        _fp0 = tts_fingerprint(_expr_spoken(seg), _voice0, args.lang, _expr_style(seg))
        if not raw_tts_fresh(raw_path, _fp0):
            try:
                voice = _voice0
                _fault("tts", cid)
                # runaway guard: indic-speak occasionally keeps generating. The TTS server already re-samples runaways
                # (TTS_RUNAWAY_RETRIES), so the pipeline adds at most ONE more try (PIPE_RUNAWAY_TRIES=2) and only when the
                # first call was quick -> no stacked retries / bounded tail latency. Keep the shortest.
                import io as _io, wave as _wave
                slot = max(seg["end"] - seg["start"], 1e-6)
                best = None
                _t0 = time.time()
                for attempt in range(max(1, int(os.environ.get("PIPE_RUNAWAY_TRIES", "2")))):      # EXACT legacy whole-slot path (also after a split failure)
                    _t0c = time.time()
                    wb = bodhan_tts(thread_session(), tts_key, _expr_spoken(seg), args.lang, voice, style=_expr_style(seg))      # the ORIGINAL legacy call
                    _record_lat(time.time() - _t0c, wb)
                    w = _wave.open(_io.BytesIO(wb))
                    d = w.getnframes() / w.getframerate()
                    if best is None or d < best[1]:
                        best = (wb, d)
                    if d / slot <= 2.5 or time.time() - _t0 > float(os.environ.get("PIPE_RUNAWAY_MAX_S", "30")):
                        break
                    log.warning("TTS runaway seg %d: %.1fs for %.1fs slot (attempt %d)", cid, d, slot, attempt + 1)
                if best[1] < 0.05:
                    raise RuntimeError("TTS returned empty audio")
                write_raw_tts(raw_path, best[0], _fp0)
            except Exception as e:
                log.error("TTS error seg %d: %s", cid, e)
                mark_fallback(seg, "tts", e)
                seg["ratio"] = 1.0
                seg["tempo"] = 1.0
                seg["overflow_s"] = 0.0
                seg["voice"] = _voice0
                for _p in (out_path, raw_path):
                    try:
                        os.remove(_p)      # a stale wav must never be mixed in place of the failed one
                    except OSError:
                        pass
                return cid

        # compute durations
        seg.pop("runaway_fix", None)
        _runaway_guard(seg, _voice0, raw_path, "whole-slot/cache")
        trim_path = tts_dir / f"{cid}_trim.wav"
        fit_src, tts_dur = _pretrim_wav(raw_path, trim_path)
        slot_dur = max(seg["end"] - seg["start"], 1e-6)
        # refit: if even max tempo (1.25x) cannot fit the room up to the next segment, rewrite shorter at the measured rate
        room = slot_dur
        for s2 in segments:
            if s2["id"] == cid + 1:
                room = max(slot_dur, s2["start"] - seg["start"])
        if tts_dur > room * 1.25 + 0.05 and not seg.get("refit") and not seg.get("edited") and not _expr_on(seg) and os.environ.get("REFIT", "1") == "1" and not _early_chunk(out_dir):
            try:
                from sunwai.pipeline.fitting.fit_stage import graphemes, guarded_rewrite, source_lang_name, confident_source_lang
                g = graphemes(seg["tgt_text"])
                budget = max(4, int(g / tts_dur * room * 1.2))
                name = next((t.get("name") for t in json.load(open(DATA_DIR / "languages.json", encoding="utf-8"))["targets"] if t["code"] == args.lang), args.lang)
                new = guarded_rewrite(seg["src_text"], seg["tgt_text"], budget, name, args.lang, __import__("sunwai.clients.llm", fromlist=["x"]).legacy_gemini_key(),
                                      source_lang_name(out_dir), tag=f"refit seg {cid}", src_lang=confident_source_lang(out_dir))
                if new:        # script + word-split + shorter + meaning guard + one retry (same as stage_fit)
                    voice = pick_voice(args.lang, seg["gender"], seg["speaker"])
                    wb = bodhan_tts(thread_session(), tts_key, new, args.lang, voice, style=_expr_style(seg))
                    tmp = tts_dir / f"{cid}_refit.wav"
                    tmp.write_bytes(wb)
                    d2 = librosa.get_duration(path=str(tmp))
                    if os.environ.get("FIT_TRIM", "1") == "1" and os.environ.get("MIX_TRIM", "1") == "1":
                        _p2, d2 = _pretrim_wav(tmp, tts_dir / f"{cid}_trim2.wav")      # same basis as tts_dur
                        try:
                            os.remove(tts_dir / f"{cid}_trim2.wav")
                        except OSError:
                            pass
                    if d2 < tts_dur:
                        log.info("refit seg %d: %.1fs -> %.1fs (room %.1fs, budget %d g)", cid, tts_dur, d2, room, budget)
                        seg["pre_refit_text"], seg["tgt_text"], seg["refit"] = seg["tgt_text"], new, True
                        for _k in ("num_text", "num_src", "num_ver", "num_lang"):
                            seg.pop(_k, None)          # v3: the refit text is spoken as written
                        write_raw_tts(raw_path, wb, tts_fingerprint(new, voice, args.lang, _expr_style(seg)))
                        os.remove(tmp) if tmp.exists() else None
                        fit_src, tts_dur = _pretrim_wav(raw_path, trim_path)
            except Exception as e:
                log.warning("refit seg %d failed: %s", cid, e)
        if _runaway_guard(seg, pick_voice(args.lang, seg["gender"], seg["speaker"]), raw_path, "refit"):
            fit_src, tts_dur = _pretrim_wav(raw_path, trim_path)
        ratio = tts_dur / slot_dur
        if ratio > 1.0:
            tempo = min(ratio, 1.25)
        elif ratio < 0.8:
            tempo = 1.0
        else:
            tempo = 1.0
        if ratio < 1.0 and not seg.get("edited"):      # dubs ended early; slow down gently (floor TTS_SLOW_MIN, default 0.9) to fill the slot
            tempo = max(ratio, _tts_slow_min())
        # apply tempo
        run_ffmpeg([
            "-i", str(fit_src),
            "-filter:a", f"atempo={tempo:.6f}",
            "-ac", "1", "-ar", "24000",
            str(out_path),
        ])
        if fit_src is not raw_path:
            try:
                os.remove(fit_src)
            except OSError:
                pass
        fitted_dur = librosa.get_duration(path=str(out_path))
        seg.pop("slack_added", None)
        if (_flag("TTS_SLACK_PAUSES") and not seg.get("edited") and ratio < 1.0 and fitted_dur < slot_dur - 0.15):
            try:                                        # the rest of the slack goes into the TTS's own phrase pauses
                import soundfile as _sf
                _y, _sr0 = _sf.read(str(out_path), dtype="float32")
                _y2, _add = _stretch_pauses(_y, _sr0, slot_dur - fitted_dur)
                if _add > 0.02 and not np.isfinite(_y2).all():
                    raise RuntimeError("slack rewrite produced non-finite samples")
                if _add > 0.02:
                    _tmp = out_path.with_name(f".{out_path.stem}.slack{os.getpid()}_{_threading.get_ident()}.wav")
                    try:
                        _sf.write(str(_tmp), _y2, _sr0, subtype="PCM_16")
                        _chk, _sr1 = _sf.read(str(_tmp), dtype="float32")            # validate: readable, finite, duration within +-5%
                        _exp = fitted_dur + _add
                        if not (np.isfinite(_chk).all() and _sr1 == _sr0 and abs(len(_chk) / _sr1 - _exp) <= 0.05 * _exp):
                            raise RuntimeError("slack rewrite failed validation")
                        os.replace(_tmp, out_path)
                        fitted_dur = len(_chk) / _sr1
                        seg["slack_added"] = round(_add, 3)
                    finally:
                        try:
                            os.remove(_tmp)
                        except OSError:
                            pass
            except Exception as _e:
                log.warning("slack pauses seg %s skipped: %s", cid, _e)
        # overflow: still longer than slot_dur
        allowed = slot_dur
        # allowed spill into following gap up to next segment start
        nxt = None
        for s in segments:
            if s["id"] == cid + 1:
                nxt = s
                break
        if nxt is not None:
            allowed = max(slot_dur, nxt["start"] - seg["start"])
        if fitted_dur > allowed + 1e-3:
            overflow_s = fitted_dur - allowed
            seg["overflow"] = True
            # truncate with 50 ms fade to allowed length
            run_ffmpeg([
                "-i", str(out_path),
                "-t", f"{allowed}",
                "-af", f"afade=t=out:st={max(0,allowed-0.05)}:d=0.05",
                "-ac", "1", "-ar", "24000",
                str(out_path),
            ])
        else:
            seg["overflow"] = False
            overflow_s = 0.0

        seg["ratio"] = ratio
        seg["tempo"] = tempo
        seg["overflow_s"] = overflow_s
        seg["voice"] = pick_voice(args.lang, seg["gender"], seg["speaker"])
        try:
            _split_upgrade(seg, _voice0, out_path, state)
        except _SplitRestoreFailed:
            raise                                               # incoherent slot: let synth() mark the fallback
        except Exception:
            pass
        return cid

    _FIT_KEYS = ("ratio", "tempo", "overflow", "overflow_s", "voice", "slack_added", "sent_pieces", "legacy_fit")

    def _split_upgrade(seg, voice, out_path, state):
        """the exact legacy whole-slot audio is ALREADY on disk. Optionally replace it by per-sentence audio when its pauses are
        off. Skip / failure / deadline / exception => legacy audio AND legacy segment fields are exactly as before. The legacy file is
        kept as a backup until the replacement and its metadata are committed; on ANY error after the replace (including
        KeyboardInterrupt/SystemExit, which are re-raised after the rollback) the audio is restored first and the fit fields only
        after a verified audio restore. If the audio cannot be restored the segment is marked as a TTS fallback (original audio kept
        by the mix) and the backup is kept: never legacy metadata over split audio, never a deleted only-copy."""
        state_ = {"got": False, "bak": None, "safe": True}   # safe: the backup may be deleted
        snap = {k: (seg[k] if k in seg else _MISSING) for k in _FIT_KEYS}

        def rollback(e):
            state_["safe"] = False
            bak = state_["bak"]
            restored = (not state.get("replaced")) or (bak is not None and bak.exists())      # a missing backup only counts as safe BEFORE the replace
            need_copy = bool(state.get("replaced"))
            if need_copy and restored:
                restored = False
                for how in (os.replace, shutil.copyfile):         # second chance: copy instead of rename
                    try:
                        how(bak, out_path); restored = True; break
                    except Exception as e2:
                        try:
                            log.error("per-sentence TTS seg %s: legacy audio restore failed (%s)", seg.get("id"), e2)
                        except Exception:
                            pass
            elif need_copy:
                restored = False                                  # replaced but no backup to restore from
            if restored:
                state_["safe"] = True
                for k, v in snap.items():
                    if v is _MISSING:
                        seg.pop(k, None)
                    else:
                        seg[k] = v
                seg["sent_skip"] = "error"
            else:                                               # audio and legacy metadata can no longer agree: coherent failed-slot state, then propagate
                for k in ("sent_pieces", "slack_added", "legacy_fit", "sent_skip"):
                    seg.pop(k, None)
                seg.update(ratio=1.0, tempo=1.0, overflow=False, overflow_s=0.0)
                try:
                    os.remove(out_path)
                except OSError:
                    pass
                try:
                    log.error("per-sentence TTS seg %s: legacy restore failed; backup kept at %s", seg.get("id"), bak)
                except Exception:
                    pass
                raise _SplitRestoreFailed(f"legacy audio restore failed: {e}")      # synth() marks the slot as a TTS fallback (original audio kept)

        try:
            if not _flag("TTS_SENT_SPLIT", "1"):
                return
            if seg.get("edited"):
                seg["sent_skip"] = "edited"; return
            if _early_chunk(out_dir):
                seg["sent_skip"] = "early_chunk"; return
            if _tts_slow_now():
                seg["sent_skip"] = "slow"; return
            if not _src_subspans(out_dir, seg):
                seg["sent_skip"] = "no_pause"; return
            state_["got"] = _split_sem().acquire(blocking=False)     # at most TTS_SENT_MAX_CONC split slots per chunk at a time
            if not state_["got"]:
                seg["sent_skip"] = "cap"; return
            bak = out_path.with_name(f".{out_path.stem}.legacy{os.getpid()}_{_threading.get_ident()}.wav")
            state_["bak"] = bak                                 # tracked before the copy so a failed/partial copy is cleaned up
            shutil.copyfile(out_path, bak)
            state_["safe"] = False                              # from here on the backup is the only way back, until commit/restore
            if _synth_pieces(seg, voice, out_path, state):
                lf = {k: snap[k] for k in ("ratio", "tempo", "overflow", "overflow_s", "slack_added") if snap[k] is not _MISSING}
                seg["legacy_fit"] = lf
                seg.pop("slack_added", None)                     # the active fields describe the split audio only
                seg.pop("sent_skip", None)
            else:
                seg["sent_skip"] = state.get("skip", "other")
            state_["safe"] = True                               # committed (or nothing was replaced)
        except Exception as e:
            try:
                log.warning("per-sentence TTS seg %s failed (%s); restoring legacy audio", seg.get("id"), e)
            except Exception:
                pass
            try:
                rollback(e)
            except _SplitRestoreFailed:
                raise
            except Exception:
                pass
        except BaseException as e:                              # KeyboardInterrupt / SystemExit: roll back, then propagate
            try:
                rollback(e)
            except Exception:
                pass
            raise
        finally:
            bak = state_["bak"]
            for fn in (lambda: state_["safe"] and bak is not None and bak.exists() and os.remove(bak),
                       lambda: state_["got"] and _split_sem().release(),
                       lambda: (state_["got"] or state.get("made")) and _safe_cleanup(seg.get("id"), tts_dir, state.get("made"))):
                try:
                    fn()
                except Exception:
                    pass

    with ThreadPoolExecutor(max_workers=_env_workers("TTS_WORKERS", max(2, 2 * len(TTS_BASES)))) as ex:
        futures = [ex.submit(synth, seg) for seg in segments if not seg["error"]]
        for fut in as_completed(futures):
            fut.result()  # propagate / log inside

    atomic_write_json(out_dir / "segments.json", segments, indent=2)
    enforce_fallback_budget(segments, "tts")
    try:
        from collections import Counter as _C
        _c = _C(("split" if s.get("sent_pieces") else s.get("sent_skip", "n/a")) for s in segments if not s.get("error"))
        log.info("Stage tts: split outcomes %s", dict(_c))
    except Exception:
        pass
    log.info("Stage tts: done")
    return segments

# ---------------------------------------------------------------------------
# Stage 9: Separation (Demucs)
# ---------------------------------------------------------------------------
def stage_separate(out_dir: Path):
    bed = out_dir / "bed.wav"
    voc = out_dir / "vocals.wav"
    if bed.exists() and voc.exists():
        log.info("Stage separate: cached")
        return

    # GPU separation service (sep_server.py): ~1 s vs 40-90 s on contended CPU. Falls back to CPU below.
    _sep = bodhan_cfg.sep_base()
    if _sep and os.environ.get("SEP_GPU", "1") == "1":
        try:
            _r = requests.post(_sep.rstrip("/") + "/separate",
                               json={"in_wav": str(out_dir / "src44k.wav"), "out_dir": str(out_dir)}, timeout=300)
            _r.raise_for_status()
            if bed.exists() and voc.exists():
                log.info("Stage separate: done (GPU service %.1fs)", _r.json().get("secs", -1))
                return
        except Exception as e:
            log.warning("GPU separation failed (%s); falling back to CPU", e)

    import torch
    import torch.multiprocessing as _mp
    os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
    torch.set_num_threads(os.cpu_count() or 1)

    import demucs.pretrained as dp
    import demucs.apply as da
    model = dp.get_model("htdemucs")
    model.to("cpu").eval()
    # apply_model signature varies; build signals
    w44 = out_dir / "src44k.wav"
    y, sr = librosa.load(str(w44), sr=44100, mono=False, dtype="float32")
    if y.ndim == 1:
        y = np.stack([y, y])
    # demucs expects (channels, time) batch dim
    sig = torch.from_numpy(y).unsqueeze(0)  # 1 x 2 x T
    with torch.no_grad():
        out = da.apply_model(model, sig, split=True, overlap=0.25, device="cpu")
    stems = out[0].cpu().numpy()  # n_stems x 2 x T
    names = list(model.sources)
    # find vocals index
    try:
        v_idx = names.index("vocals")
    except ValueError:
        v_idx = len(names) - 1
    voc_np = stems[v_idx]                 # 2 x T
    bed_np = stems.sum(axis=0) - voc_np   # sum(non-vocal)

    save_wav_stereo(voc, voc_np.T, 44100)
    save_wav_stereo(bed, bed_np.T, 44100)
    log.info("Stage separate: done")


# ---------------------------------------------------------------------------
# speech-onset alignment helpers (trim TTS lead/trail silence, per-pause re-layout)
# ---------------------------------------------------------------------------
def _frame_db(y: np.ndarray, sr: int, hop_s: float = 0.01) -> np.ndarray:
    hop = max(1, int(hop_s * sr))
    n = len(y) // hop
    if n == 0:
        return np.zeros(0)
    f = y[:n * hop].reshape(n, hop)
    return 10.0 * np.log10(np.mean(f * f, axis=1) + 1e-10)


def _speech_bounds(db: np.ndarray, rel: float, need: int = 4):
    """(first, last_exclusive, active_mask) frame indices of sustained speech; threshold = p95 - rel dB, floor -70 dB."""
    if len(db) < need:
        return None, None, None
    act = db > max(float(np.percentile(db, 95)) - rel, -70.0)
    on = off = None
    for i in range(len(act) - need + 1):
        if act[i:i + need].all():
            on = i
            break
    for i in range(len(act), need - 1, -1):
        if act[i - need:i].all():
            off = i
            break
    if on is None or off is None or off <= on:
        return None, None, None
    return on, off, act


MIX_TRIM_REL = _envf("MIX_TRIM_REL", 35, 10.0, 80.0)      # conservative: only cuts audio >35 dB below the p95 level
MIX_KEEP_LEAD = _envf("MIX_KEEP_LEAD", 0.03, 0.0, 0.5)  # s of audio kept before the detected onset
MIX_KEEP_TAIL = _envf("MIX_KEEP_TAIL", 0.05, 0.0, 1.0)


def _pretrim_wav(raw_path, dst):
    """trim lead/trail silence BEFORE duration fitting (the mix trims anyway, so fitting/truncating the untrimmed
    audio over-speeds or clips the last word). Returns (path_to_fit, duration_s). The raw TTS cache is never modified; when
    the trim is not clearly safe (kill switch FIT_TRIM=0 / MIX_TRIM=0, no sustained speech found, nothing to cut, or more than
    half of the audio would be cut) the untrimmed raw file is used unchanged."""
    raw_dur = librosa.get_duration(path=str(raw_path))
    if os.environ.get("FIT_TRIM", "1") != "1" or os.environ.get("MIX_TRIM", "1") != "1":
        return raw_path, raw_dur
    try:
        t, sr0 = librosa.load(str(raw_path), sr=None, mono=True, dtype="float32")
        if len(t) == 0:
            return raw_path, raw_dur
        tt, _lead = _trim_tts(t, sr0)
        if len(tt) >= len(t) - int(0.01 * sr0) or len(tt) < 0.5 * len(t):
            return raw_path, raw_dur
        import soundfile as _sf
        _sf.write(str(dst), tt, sr0, subtype="PCM_16")
        return dst, len(tt) / sr0
    except Exception as _e:
        log.warning("pretrim skipped (%s): %s", raw_path, _e)
        return raw_path, raw_dur


def _peak_limit(x: np.ndarray, ceiling: float, sr: int, look_ms: float, rel_ms: float) -> np.ndarray:
    """sample-accurate look-ahead TRUE-peak limiter (x = T x C; 4x oversampled detection). Offline, so the look-ahead needs no latency: the gain
    at sample n is a centred +-look window minimum of the required gain, smoothed over the same window (so it is always
    <= the required gain at n, i.e. |y| <= ceiling), followed by a one-pole release (gain recovers ~90% in rel_ms).
    Chunks whose peak is already <= ceiling are returned unchanged; gain reduction is local to the overs."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    p = np.max(np.abs(x), axis=1)
    if not (float(p.max()) * 1.5 > ceiling):          # cannot reach the ceiling even with a 3.5 dB inter-sample over
        return x
    # inter-sample (true) peak: 4x oversampled |x|, max-pooled back onto the original sample grid
    from scipy.signal import resample_poly
    p = np.max(np.abs(resample_poly(x.astype(np.float32, copy=False), 4, 1, axis=0)), axis=1)[:4 * len(x)]
    p = p.reshape(len(x), 4).max(axis=1)
    if not (float(p.max()) > ceiling):
        return x
    g = np.minimum(1.0, ceiling / np.maximum(p, 1e-12)).astype(np.float64)
    L = max(1, int(look_ms * 1e-3 * sr))
    ga = uniform_filter1d(minimum_filter1d(g, size=2 * L + 1, mode="nearest"), size=2 * L + 1, mode="nearest")
    ga = np.minimum(ga, g)
    a = math.exp(-math.log(10.0) / max(1.0, rel_ms * 1e-3 * sr))
    n_tot = len(ga)
    act = np.flatnonzero(ga < 1.0)
    r = np.ones(n_tot, dtype=np.float64)
    gl = ga.tolist()
    prev = 1.0
    n = int(act[0])
    while n < n_tot:
        v = 1.0 - (1.0 - prev) * a
        gv = gl[n]
        if gv < v:
            v = gv
        r[n] = v
        prev = v
        n += 1
        if 1.0 - prev < 1e-6 and n < n_tot and gl[n] >= 1.0:
            k = int(np.searchsorted(act, n))
            if k >= len(act):
                break
            n = int(act[k])
            prev = 1.0
    y = x * r[:, None].astype(x.dtype)
    return np.clip(y, -ceiling, ceiling).astype(x.dtype)


def _trim_tts(t: np.ndarray, sr: int):
    """Trim leading/trailing silence of one TTS wav. Returns (trimmed, lead_kept_s). No-op when nothing meaningful to cut."""
    db = _frame_db(t, sr)
    on, off, _ = _speech_bounds(db, MIX_TRIM_REL)
    if on is None:
        return t, 0.0
    a = max(0, int((on * 0.01 - MIX_KEEP_LEAD) * sr))
    b = min(len(t), int((off * 0.01 + MIX_KEEP_TAIL) * sr))
    if a < int(0.01 * sr) and b > len(t) - int(0.01 * sr):
        return t, MIX_KEEP_LEAD if on * 0.01 >= MIX_KEEP_LEAD else on * 0.01
    out = t[a:b].copy()
    fl = min(int(0.005 * sr), len(out) // 4)           # 5 ms edge fades: no click at the cut
    if fl > 0:
        if a > 0:
            out[:fl] *= np.linspace(0.0, 1.0, fl, dtype=out.dtype)
        if b < len(t):
            out[-fl:] *= np.linspace(1.0, 0.0, fl, dtype=out.dtype)
    return out, (on * 0.01 - a / sr)


def _orig_sentence_resumes(vad_path: Path, s: float, e: float, min_gap: float = 0.35):
    """Chunk-time speech-resume instants after every >= min_gap VAD gap inside [s, e]."""
    try:
        iv = json.load(open(vad_path, encoding="utf-8"))
        iv = [((v["start"], v["end"]) if isinstance(v, dict) else tuple(v)) for v in iv]
    except Exception:
        return []
    iv = [(a, b) for a, b in iv if b > s - 0.01 and a < e + 0.01]
    return [a2 for (a1, b1), (a2, b2) in zip(iv, iv[1:]) if a2 - b1 > min_gap]


MIX_LIMITER = os.environ.get("MIX_LIMITER", "1") == "1"            # (0 = old whole-chunk peak scaling)
MIX_LIM_LOOKAHEAD_MS = _envf("MIX_LIM_LOOKAHEAD_MS", 5.0, 0.5, 20.0)
MIX_LIM_RELEASE_MS = _envf("MIX_LIM_RELEASE_MS", 80.0, 10.0, 500.0)   # time to ~90% gain recovery
MIX_LEVEL_FIX = os.environ.get("MIX_LEVEL_FIX", "1") == "1"        # (0 = old loudness match)
MIX_ALIGN_PAUSES = os.environ.get("MIX_ALIGN_PAUSES", "1") == "1"   # default ON (guarded, fail-open, capped by next-slot room)
MIX_ALIGN_MIN_DRIFT = _envf("MIX_ALIGN_MIN_DRIFT", 0.25, 0.05, 5.0)


def _align_pauses(t: np.ndarray, sr: int, lead: float, resumes_rel, slot_dur: float, room: float = None):
    """Re-lay a (trimmed) multi-sentence TTS wav so each sentence restarts where the original speaker resumed.

    t starts `lead` s before its detected onset; resumes_rel = original resume instants relative to the original onset.
    Cuts t inside its N longest internal silences (>= 0.2 s) and re-places piece k at resumes_rel[k-1] (never closer than
    0.15 s after the previous piece). Returns (new_t, info) or (t, None) when the guards fail (counts differ, relative
    positions disagree, drift already small, or the result would exceed the slot)."""
    N = len(resumes_rel)
    if N == 0:
        return t, None
    db = _frame_db(t, sr)
    on, off, act = _speech_bounds(db, 30.0)
    if on is None:
        return t, None
    # internal silence runs
    runs, i = [], on
    while i < off:
        if not act[i]:
            j = i
            while j < off and not act[j]:
                j += 1
            if j - i >= 20:
                runs.append((i, j))
            i = j
        else:
            i += 1
    if len(runs) < N:
        return t, None
    runs = sorted(sorted(runs, key=lambda r: -(r[1] - r[0]))[:N])
    d_span = (off - on) * 0.01
    dub_pos = [((r[0] + r[1]) / 2 - on) * 0.01 for r in runs]            # dub pause centre, relative to dub onset
    # guard: relative positions must agree (translation kept the sentence order/shape)
    o_pos = [r - 0.2 for r in resumes_rel]                                 # approx. original pause centre
    if any(abs(a / max(slot_dur, 1e-3) - b / max(d_span, 1e-3)) > 0.3 for a, b in zip(o_pos, dub_pos)):
        return t, None
    drift = [r - ((rn[1] * 0.01) - on * 0.01) for r, rn in zip(resumes_rel, runs)]   # orig resume - dub resume
    if max(abs(d) for d in drift) < MIX_ALIGN_MIN_DRIFT:
        return t, None
    # build pieces: piece 0 = [0 .. runs[0] start], piece k = [runs[k-1] end .. runs[k] start], last = [.. end]
    cuts = [(a * 0.01, b * 0.01) for a, b in runs]
    pieces = []
    prev = 0.0
    for (ca, cb) in cuts:
        pieces.append((prev, ca)); prev = cb
    pieces.append((prev, off * 0.01 + MIX_KEEP_TAIL))
    out_parts, cur = [], 0.0       # cur = write head (s) relative to wav start (which sits `lead` before the onset)
    placed = []
    for k, (a, b) in enumerate(pieces):
        seg = t[int(a * sr):int(b * sr)]
        if k == 0:
            start_at = 0.0
        else:
            start_at = max(cur + 0.15, lead + resumes_rel[k - 1])     # resume instants are relative to the onset
        out_parts.append((start_at, seg))
        cur = start_at + len(seg) / sr
        placed.append(start_at)
    total = cur
    if total > max(len(t) / sr, slot_dur + lead) + 0.3:
        return t, None
    if room is not None and total > max(len(t) / sr, room + lead):   # never run into the next slot's start
        return t, None
    new = np.zeros(int(total * sr) + 1, dtype=t.dtype)
    fl = int(0.004 * sr)
    for start_at, seg in out_parts:
        seg = seg.copy()
        if len(seg) > 2 * fl:
            seg[:fl] *= np.linspace(0, 1, fl, dtype=seg.dtype); seg[-fl:] *= np.linspace(1, 0, fl, dtype=seg.dtype)
        i0 = int(start_at * sr)
        new[i0:i0 + len(seg)] += seg[:len(new) - i0]
    return new, dict(n_pauses=N, drift_before=[round(d, 3) for d in drift])

# ---------------------------------------------------------------------------
# Stage 10: Mix + mux
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# (1) keep non-speech vocal events (laughter, breath, singing) that the dub does not cover, (2) bounded smooth
# ducking of the separated bed under the dub's speech, (3) keep the ORIGINAL audio in planner overlap spans.
# Every piece has an env kill switch (parsed robustly: 0/false/off/no -> off) and fails open to the old mix.
# ---------------------------------------------------------------------------
def _g90c_flag(name: str, default: bool = True) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if v in ("0", "false", "off", "no"):
        return False
    if v in ("1", "true", "on", "yes"):
        return True
    return default


MIX_KEEP_EVENTS = _g90c_flag("MIX_KEEP_EVENTS", False)   # default OFF; set 1 to enable
MIX_EVENT_THR = _envf("MIX_EVENT_THR", 0.50, 0.05, 0.99)              # AudioSet event-class probability needed to keep a vocal event
MIX_EVENT_MIN_S = _envf("MIX_EVENT_MIN_S", 0.20, 0.05, 2.0)
MIX_EVENT_MAX_CAND = int(_envf("MIX_EVENT_MAX_CAND", 12, 1, 100))
MIX_DUCK = _g90c_flag("MIX_DUCK", True)
MIX_DUCK_DB = _envf("MIX_DUCK_DB", 3.0, 0.0, 6.0)                     # bed reduction under dub speech (bounded to 6 dB)
MIX_DUCK_ATTACK_MS = _envf("MIX_DUCK_ATTACK_MS", 60.0, 5.0, 500.0)    # gain-fall time constant (look-ahead = 2x so the bed is down BEFORE the onset)
MIX_DUCK_RELEASE_MS = _envf("MIX_DUCK_RELEASE_MS", 400.0, 20.0, 3000.0)
MIX_DUCK_HOLD_MS = _envf("MIX_DUCK_HOLD_MS", 300.0, 0.0, 2000.0)      # speech-activity gaps shorter than this do not release the duck (no pumping)
MIX_KEEP_OVERLAP = _g90c_flag("MIX_KEEP_OVERLAP", True)
MIX_OVERLAP_MAX_CUT = _envf("MIX_OVERLAP_MAX_CUT", 0.5, 0.0, 1.0)     # never cut more than this fraction of one segment's dub speech
_G90C_HOP = 0.01
_AST = {}


def _dub_activity_frames(t: np.ndarray, sr: int, rel: float = 30.0) -> np.ndarray:
    """bool per 10 ms frame: dub speech present (frame power > p95 - rel dB, floor -70 dB). All-silent input -> all False."""
    db = _frame_db(np.asarray(t, dtype=np.float32), sr, _G90C_HOP)
    if len(db) == 0:
        return np.zeros(0, dtype=bool)
    return db > max(float(np.percentile(db, 95)) - rel, -70.0)


def _g90c_activity(dubbed, N: int, sr: int) -> np.ndarray:
    """global per-frame dub-speech activity (union over placed segments)."""
    hop = int(_G90C_HOP * sr)
    act = np.zeros(N // hop + 1, dtype=bool)
    for (s, em, t_st) in dubbed:
        a = _dub_activity_frames(t_st[:, 0], sr)
        f0 = s // hop
        n = min(len(a), len(act) - f0)
        if n > 0:
            act[f0:f0 + n] |= a[:n]
    return act


def _duck_gain(act: np.ndarray, depth_db: float, attack_ms: float, release_ms: float, hold_ms: float) -> np.ndarray:
    """Per-frame bed gain (<= 1). act -> hold (forward, after each real offset) -> onset look-ahead (2 x attack, backward only) -> one-pole attack/release.
    Depth is bounded to 6 dB; depth 0 or an all-False activity returns exactly 1.0 everywhere."""
    from scipy.ndimage import binary_closing
    depth_db = max(0.0, min(6.0, float(depth_db)))
    n = len(act)
    if n == 0 or depth_db <= 0.0 or not act.any():
        return np.ones(n, dtype=np.float64)
    hop_ms = _G90C_HOP * 1000.0
    k = int(round(hold_ms / hop_ms))
    la = int(round(2.0 * attack_ms / hop_ms))
    a = act.astype(np.int32)
    c = np.concatenate([[0], np.cumsum(a)])
    idx = np.arange(n)
    # hold: forward dilation (bridges gaps <= hold AND keeps the duck for `hold` after the real speech offset)
    lo = np.maximum(0, idx - k)
    a = (c[idx + 1] - c[lo]) > 0
    # look-ahead: BACKWARD dilation only (earlier onsets); real offsets are never moved earlier
    c = np.concatenate([[0], np.cumsum(a.astype(np.int32))])
    hi = np.minimum(n, idx + la + 1)
    a = (c[hi] - c[idx]) > 0
    gmin = 10.0 ** (-depth_db / 20.0)
    tgt = np.where(a, gmin, 1.0)
    ka = math.exp(-_G90C_HOP / (attack_ms * 1e-3))
    kr = math.exp(-_G90C_HOP / (release_ms * 1e-3))
    g = np.empty(n, dtype=np.float64)
    prev = 1.0
    for i in range(n):
        c = ka if tgt[i] < prev else kr
        prev = tgt[i] + (prev - tgt[i]) * c
        g[i] = prev
    return g


def _frames_to_samples(g: np.ndarray, N: int, sr: int) -> np.ndarray:
    hop = int(_G90C_HOP * sr)
    xc = (np.arange(len(g)) + 0.5) * hop
    return np.interp(np.arange(N), xc, g).astype(np.float32)


def _g90c_overlap_spans(out_dir: Path, N: int, sr: int):
    """Chunk-time (a, b) sample spans from the planner's <job>/overlap_spans.json (seconds on the job timeline,
    {"spans": [{"start","end","kind"}]}). Missing/unreadable/malformed file -> []."""
    try:
        cands = []
        sm = os.environ.get("SPEAKER_MAP")
        if sm:
            cands.append(Path(os.path.dirname(sm)))
        cands.append(Path(out_dir).resolve().parent.parent)
        doc = None
        for jd in cands:
            p = jd / "overlap_spans.json"
            if p.exists():
                doc = json.load(open(p, encoding="utf-8"))
                break
        if not isinstance(doc, dict):
            return []
        try:
            c0 = float(os.environ.get("CHUNK_START", ""))
        except Exception:
            try:
                c0 = float(json.load(open(Path(out_dir) / "meta.json", encoding="utf-8")).get("start", 0.0))
            except Exception:
                return []                                    # unknown chunk offset: do nothing rather than guess
        kinds = {k.strip() for k in os.environ.get("MIX_OVERLAP_KINDS", "overlap,short_turn").split(",") if k.strip()}
        out = []
        for sp in doc.get("spans", []):
            a, b = float(sp["start"]) - c0, float(sp["end"]) - c0
            if not (math.isfinite(a) and math.isfinite(b)) or b <= a or sp.get("kind", "overlap") not in kinds:
                continue
            a, b = max(0, int(a * sr)), min(N, int(b * sr))
            if b - a > int(0.05 * sr):
                out.append((a, b))
        return out
    except Exception as e:
        log.warning("overlap_spans ignored: %s", e)
        return []


def _g90c_guard_overlap(spans, dubbed, sr: int, max_cut: float):
    """accept spans one by one; a span is dropped if the UNION of accepted spans + this one would remove more than max_cut of
    any single dubbed segment's speech frames."""
    hop = int(_G90C_HOP * sr)
    segs = []
    for (s, em, t_st) in dubbed:
        act = _dub_activity_frames(t_st[:, 0], sr)
        if act.any():
            segs.append((np.arange(len(act)) * hop + s, act, float(act.sum())))
    keep = []
    for (a, b) in spans:
        trial = keep + [(a, b)]
        ok = True
        for (f, act, tot) in segs:
            cov = np.zeros(len(f), dtype=bool)
            for (x, y) in trial:
                cov |= (f >= x) & (f < y)
            if float(np.sum(act & cov)) / tot > max_cut:
                ok = False
                break
        if ok:
            keep.append((a, b))
        else:
            log.info("mix: overlap span %.2f-%.2f s skipped (union would cut > %.0f%% of a dubbed segment)", a / sr, b / sr, 100 * max_cut)
    return keep


def _ast_probs(x16: np.ndarray):
    """AudioSet-AST class probabilities for one 16 kHz mono clip, or None when the model is unavailable (fail-open)."""
    try:
        if "m" not in _AST:
            import glob as _glob
            from transformers import ASTFeatureExtractor, ASTForAudioClassification
            hubs = _glob.glob(os.path.expanduser(os.environ.get("MIX_AST_DIR", "~/.cache/huggingface/hub/models--MIT--ast-finetuned-audioset-10-10-0.4593/snapshots/*")))
            if not hubs:
                raise RuntimeError("no AST snapshot")
            import torch
            torch.set_num_threads(min(4, os.cpu_count() or 1))        # do not starve the co-located model servers
            _AST["fe"] = ASTFeatureExtractor.from_pretrained(hubs[0], local_files_only=True)
            _AST["m"] = ASTForAudioClassification.from_pretrained(hubs[0], local_files_only=True).eval()
            _AST["lab"] = [_AST["m"].config.id2label[i] for i in range(len(_AST["m"].config.id2label))]
        import torch
        inp = _AST["fe"]([x16], sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            p = torch.sigmoid(_AST["m"](**inp).logits)[0].numpy()
        return dict(zip(_AST["lab"], p.tolist()))
    except Exception as e:
        if not _AST.get("warned"):
            log.warning("event detector unavailable (%s); non-speech vocal events not kept", e)
            _AST["warned"] = True
        return None


_EVENT_CLASSES = ("Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle", "Breathing", "Gasp", "Sigh", "Singing", "Humming",
                  "Crying, sobbing", "Cough", "Sneeze", "Groan", "Yawn", "Chant", "Choir")


def _g90c_event_spans(orig, vocals, replaced, dub_act, N: int, sr: int):
    """(a, b) sample spans inside the replaced region, NOT covered by dub speech (+120 ms), where the separated vocals are active and
    the AudioSet detector sees a non-speech vocal event on the original mix. Returns (spans, info)."""
    from scipy.ndimage import binary_dilation
    from scipy.signal import resample_poly
    hop = int(_G90C_HOP * sr)
    nf = N // hop
    if nf < 10:
        return [], {}
    rep_f = replaced[:nf * hop].reshape(nf, hop).mean(axis=1) > 0.5
    da = np.zeros(nf, dtype=bool)
    da[:min(nf, len(dub_act))] = dub_act[:nf]
    free = rep_f & ~binary_dilation(da, iterations=12)
    if not free.any():
        return [], {"cand": 0}
    vm = vocals.mean(axis=1).astype(np.float32)
    vdb = _frame_db(vm, sr, _G90C_HOP)[:nf]
    vact = vdb > max(float(np.percentile(vdb, 95)) - 30.0, -60.0)
    cand = free & vact
    # runs >= MIX_EVENT_MIN_S, merging gaps <= 100 ms
    cand = binary_dilation(cand, iterations=5) & free
    runs, i = [], 0
    while i < nf:
        if cand[i]:
            j = i
            while j < nf and cand[j]:
                j += 1
            if (j - i) * _G90C_HOP >= MIX_EVENT_MIN_S:
                runs.append((i, j))
            i = j
        else:
            i += 1
    spans, info = [], {"cand": len(runs), "kept": 0, "probs": []}
    if not runs:
        return [], info
    # classify the separated VOCALS of the candidate run itself (zero-padded to >= 1 s; no neighbouring dub-slot speech, no bed music)
    vm16 = vm
    runs = sorted(runs, key=lambda r: r[0] - r[1])[:MIX_EVENT_MAX_CAND]       # longest first, bounded detector cost per chunk
    for (i, j) in sorted(runs):
        a0, b0 = i * hop, j * hop
        x16 = resample_poly(vm16[a0:b0], 160, 441).astype(np.float32)
        if len(x16) < 16000:
            x16 = np.pad(x16, (0, 16000 - len(x16)))
        pr = _ast_probs(x16)
        if pr is None:
            return [], info
        ev = max((pr.get(k, 0.0), k) for k in _EVENT_CLASSES)
        sp = pr.get("Speech", 0.0)
        info["probs"].append([round(i * _G90C_HOP, 2), round(j * _G90C_HOP, 2), ev[1], round(ev[0], 3), round(sp, 3)])
        if ev[0] >= MIX_EVENT_THR and ev[0] >= 0.5 * sp:
            spans.append((a0, b0))
            info["kept"] += 1
    return spans, info


def _g90c_span_mask(spans, N: int, sr: int, fade_s: float = 0.02) -> np.ndarray:
    """0..1 mask with short (fade_s) Hann edges for kept events."""
    from scipy.signal.windows import hann
    cf = max(1, int(fade_s * sr))
    mask = np.zeros(N, dtype=np.float32)
    for a, b in spans:
        mask[max(0, a):min(N, b)] = 1.0
    win = hann(2 * cf + 1, sym=False)
    return np.clip(np.convolve(mask, win, mode="same") / win.sum(), 0.0, 1.0).astype(np.float32)


def _chunk_owned_samples(out_dir: Path, N: int, sr: int):
    """returns (samples, exact, reason). exact=True only when the chunk's meta.json yields a plausible owned length
    = round(end*sr) - round(start*sr) (cumulative rounding: chunks tile the source with 0 missing/duplicated samples).
    CHUNK_EXACT_OWN=0 (the single kill switch; legacy alias MUX_FULL_AUDIO=0), missing/invalid/implausible meta -> (N, False, reason)
    and the caller takes the legacy -shortest path."""
    off = ("0", "false", "off", "no")
    if os.environ.get("CHUNK_EXACT_OWN", "1").strip().lower() in off or os.environ.get("MUX_FULL_AUDIO", "1").strip().lower() in off:
        return N, False, "disabled by CHUNK_EXACT_OWN=0"
    try:
        m = json.load(open(out_dir / "meta.json", encoding="utf-8"))
        st, du = float(m["start"]), float(m["dur"])
        if not (math.isfinite(st) and math.isfinite(du) and st >= 0 and du > 0):
            return N, False, f"invalid meta start/dur ({st}, {du})"
        n = int(round((st + du) * sr)) - int(round(st * sr))
        if not (0 < n <= N and N - n <= int(0.25 * sr)):
            return N, False, f"implausible owned length {n} vs {N} samples"
        return n, True, None
    except Exception as e:
        return N, False, f"meta.json unusable ({type(e).__name__}: {e})"


def _audio_track_dur(path: Path) -> Optional[float]:
    try:
        import subprocess as _sp
        o = _sp.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=duration", "-of", "csv=p=0", str(path)],
                    capture_output=True, text=True, timeout=30).stdout.strip()
        return float(o)
    except Exception:
        return None


def stage_mix(args, out_dir: Path, segments: List[Dict[str, Any]]):
    out_wav = out_dir / "dub.wav"
    out_mp4 = out_dir / f"dub_{args.lang}.mp4"

    import torch
    # Load original 44.1k stereo mix, bed, vocals
    orig, _ = librosa.load(str(out_dir / "src44k.wav"), sr=44100, mono=False, dtype="float32")
    if orig.ndim == 1:
        orig = np.stack([orig, orig])
    orig = orig.T  # T x 2
    N = orig.shape[0]

    if (out_dir / "bed.wav").exists():
        bed, _ = librosa.load(str(out_dir / "bed.wav"), sr=44100, mono=False, dtype="float32")
        bed = bed.T if bed.ndim == 2 else np.stack([bed, bed], axis=1)
    else:
        bed = orig.copy()
    if (out_dir / "vocals.wav").exists():
        vocals, _ = librosa.load(str(out_dir / "vocals.wav"), sr=44100, mono=False, dtype="float32")
        vocals = vocals.T if vocals.ndim == 2 else np.stack([vocals, vocals], axis=1)
    else:
        vocals = orig.copy()

    bed = bed[:N] if len(bed) >= N else np.pad(bed, ((0, N - len(bed)), (0, 0)))
    vocals = vocals[:N] if len(vocals) >= N else np.pad(vocals, ((0, N - len(vocals)), (0, 0)))

    sr = 44100
    cf = int(0.08 * sr)  # 80 ms crossfade samples

    def rms(x):
        return float(np.sqrt(np.mean(np.sum(x * x, axis=1)) + 1e-12))

    # Load every usable TTS wav ONCE. A segment is "dubbed" only if it has a TTS wav and is not a fallback; fallback /
    # failed segments keep the ORIGINAL audio (they are not in the mask), so their vocals are never silently removed.
    dubbed = []   # (s, e_mask, t_stereo_scaled)
    for seg in segments:
        if seg["error"] or seg.get("fallback"):
            continue
        s = max(0, min(int(seg["start"] * sr), N))
        e = max(0, min(int(seg["end"] * sr), N))
        if e - s <= 0:
            continue
        tts_path = out_dir / "tts" / f"{seg['id']}.wav"
        if not tts_path.exists():
            continue
        t, _ = librosa.load(str(tts_path), sr=44100, mono=True, dtype="float32")
        if len(t) == 0:
            continue
        # put the dub's speech onset on the original onset (trim TTS lead/trail silence, keep ~30 ms) and, for
        # multi-sentence slots, restart each sentence where the original speaker resumed. Guarded, fail-open.
        if MIX_ALIGN_PAUSES or os.environ.get("MIX_TRIM", "1") == "1":
            try:
                if os.environ.get("MIX_TRIM", "1") == "1" and not seg.get("sent_pieces"):      # pieces carry intentional silence between sentences
                    t, lead_kept = _trim_tts(t, sr)
                else:
                    lead_kept = 0.0
                if MIX_ALIGN_PAUSES and not seg.get("sent_pieces") and not seg.get("slack_added"):      # already source-placed / slack-padded in stage_tts
                    _nx = [x["start"] for x in segments if x["start"] > seg["start"] + 1e-6]
                    _room = (min(_nx) - seg["start"]) if _nx else None
                    rs = _orig_sentence_resumes(out_dir / "vad.json", seg["start"], seg["end"])
                    if rs:
                        t2, info = _align_pauses(t, sr, lead_kept, [r - seg["start"] for r in rs], seg["end"] - seg["start"], room=_room)
                        if info:
                            log.info("mix align seg %s: %s", seg["id"], info)
                            t = t2
                s = max(0, s - int(lead_kept * sr))
            except Exception as _e:
                log.warning("mix align seg %s skipped: %s", seg.get("id"), _e)
                t, _ = librosa.load(str(tts_path), sr=44100, mono=True, dtype="float32")
                s = max(0, min(int(seg["start"] * sr), N))
        ref_rms = rms(vocals[s:e])
        t_rms = float(np.sqrt(np.mean(t * t) + 1e-12))
        if MIX_LEVEL_FIX:
            ref_rms = ref_rms / math.sqrt(vocals.shape[1])     # per-channel power, same convention as the mono t_rms
        if MIX_LEVEL_FIX:        # validity on UNFLOORED power (the +1e-12 floor made rms() > 1e-8 always true)
            _vs = vocals[s:e]
            _rp = float(np.mean(np.sum(_vs * _vs, axis=1))) / vocals.shape[1] if len(_vs) else 0.0
            _valid = _rp > 1e-16 and float(np.mean(t * t)) > 1e-16
        else:
            _valid = ref_rms > 1e-8 and t_rms > 1e-8
        if _valid:
            scale = ref_rms / t_rms
            if MIX_LEVEL_FIX:                                   # real +-9 dB gain clamp (the old expression always contained `scale`)
                scale = max(10 ** (-9 / 20), min(10 ** (9 / 20), scale))
            else:
                scale = max(ref_rms / (t_rms * (10 ** (9/20))), min(ref_rms / (t_rms * (10 ** (-9/20))), scale))
            t = t * scale
        t_st = np.stack([t, t], axis=1)
        wl = min(len(t_st), N - s)
        # borrowed following silence: the TTS may run past the slot end -> extend the replaced region to the end of the
        # speech (+ one crossfade so the tail is not half-attenuated by the smoothing) instead of attenuating it away
        e_mask = max(e, s + wl)
        if s + wl > e - cf:
            e_mask = min(N, max(e, s + wl + cf))
        dubbed.append((s, e_mask, t_st[:wl]))

    mask_sm = mix_mask([(s, em) for (s, em, _) in dubbed], N, sr)
    # (all guarded, fail-open to the old mix; kill switches MIX_KEEP_OVERLAP / MIX_DUCK / MIX_KEEP_EVENTS)
    _g90 = {}
    dub_act = None
    if dubbed and (MIX_KEEP_OVERLAP or MIX_DUCK or MIX_KEEP_EVENTS):
        try:
            dub_act = _g90c_activity(dubbed, N, sr)
        except Exception as _e:
            log.warning("mix g90c activity skipped: %s", _e)
    if MIX_KEEP_OVERLAP and dubbed:
        try:
            _ovs = _g90c_overlap_spans(out_dir, N, sr)
            if _ovs:
                _ovs = _g90c_guard_overlap(_ovs, dubbed, sr, MIX_OVERLAP_MAX_CUT)
            if _ovs:
                _ovm = mix_mask(_ovs, N, sr)
                mask_sm = (mask_sm * (1.0 - _ovm)).astype(np.float32)
                _g90["overlap_kept_s"] = round(sum(b - a for a, b in _ovs) / sr, 3)
        except Exception as _e:
            log.warning("mix overlap keep skipped: %s", _e)
    base_with_bed = bed
    if MIX_DUCK and dub_act is not None and dub_act.any():
        try:
            _gf = _duck_gain(dub_act, MIX_DUCK_DB, MIX_DUCK_ATTACK_MS, MIX_DUCK_RELEASE_MS, MIX_DUCK_HOLD_MS)
            _gs = _frames_to_samples(_gf, N, sr)
            base_with_bed = bed * _gs[:, None]
            _g90["duck_min_db"] = round(float(20 * np.log10(max(float(_gs.min()), 1e-6))), 2)
        except Exception as _e:
            log.warning("mix ducking skipped: %s", _e)
            base_with_bed = bed
    _kev = None
    if MIX_KEEP_EVENTS and dub_act is not None and (out_dir / "vocals.wav").exists():
        try:
            _evs, _evi = _g90c_event_spans(orig, vocals, mask_sm, dub_act, N, sr)
            _g90["events"] = _evi
            if _evs:
                _kev = _g90c_span_mask(_evs, N, sr)
                _g90["event_kept_s"] = round(sum(b - a for a, b in _evs) / sr, 3)
        except Exception as _e:
            log.warning("mix event keep skipped: %s", _e)
    final = orig.copy()
    for i in range(0, N, 4096):
        j = min(i + 4096, N)
        m = mask_sm[i:j]
        final[i:j] = final[i:j] * (1.0 - m[:, None]) + base_with_bed[i:j] * m[:, None]
        if _kev is not None:
            final[i:j] += vocals[i:j] * (_kev[i:j] * m)[:, None]
    if _g90:
        try:
            atomic_write_json(out_dir / "g90c.json", _g90)
        except Exception as _e:
            log.warning("g90c telemetry skipped: %s", _e)
    for (s, em, t_st) in dubbed:
        wl = len(t_st)
        m = mask_sm[s:s + wl]
        final[s:s + wl] += t_st * m[:, None]
    try:                                             # cheap per-chunk telemetry (placement overlap, pre-limiter peak), always on
        _sp = sorted((s, s + len(t_st)) for (s, em, t_st) in dubbed)
        _ov = [max(0, a_end - b_s) / sr for (a_s, a_end), (b_s, b_e) in zip(_sp, _sp[1:])]
        _fin = lambda v: float(v) if np.isfinite(v) else 0.0
        atomic_write_json(out_dir / "tts_placed.json", {
            "peak_pre_limit": _fin(np.max(np.abs(final))), "frac_over_0.89": _fin(np.mean(np.abs(final) > 0.89)),
            "overlap_s": _fin(sum(_ov)), "n_overlaps_gt_50ms": int(sum(1 for o in _ov if o > 0.05)),
            "max_overlap_s": _fin(max(_ov) if _ov else 0.0), "n_dubbed": len(dubbed)})
    except Exception as _e:
        log.warning("mix telemetry skipped: %s", _e)
    if os.environ.get("MIX_DUMP_STEM") == "1":      # measurement aid: placed TTS only; never fails the mix
        try:
            _stem = np.zeros_like(final)
            for (s, em, t_st) in dubbed:
                _stem[s:s + len(t_st)] += t_st
            _tmp = out_dir / f".stemtmp_{os.getpid()}.wav"
            save_wav_stereo(_tmp, _stem, 44100)
            os.replace(_tmp, out_dir / "tts_placed.wav")
        except Exception as _e:
            log.warning("stem dump skipped: %s", _e)

    # Peak limit -1 dBFS
    ceiling = 10 ** (-1 / 20)
    if MIX_LIMITER:
        if not np.isfinite(final).all():
            log.warning("mix: non-finite samples in the mix; zeroed")
            final = np.nan_to_num(final, nan=0.0, posinf=0.0, neginf=0.0)
        final = _peak_limit(final, ceiling, sr, MIX_LIM_LOOKAHEAD_MS, MIX_LIM_RELEASE_MS)
    else:
        peak = float(np.max(np.abs(final))) + 1e-12
        if peak > ceiling:
            final = final * (ceiling / peak)

    # Atomic publish: write temp files (with the right extension for ffmpeg), validate, then os.replace.
    # The mp4 is replaced LAST so its mtime is the newest (the lip-sync worker keys on it).
    tag = f"{os.getpid()}_{_threading.get_ident()}"
    tmp_wav = out_dir / f".dubtmp_{tag}.wav"
    tmp_mp4 = out_dir / f".dubtmp_{args.lang}_{tag}.mp4"
    try:
        # . Exact mode: dub.wav = the samples this chunk owns, mux WITHOUT `-shortest` (the old mux dropped the last 2-4 AAC
        # frames, 46-186 ms, of every chunk). The audio edit list then ends exactly at the owned length. Output is validated; any
        # mismatch/exception -> the legacy path (-shortest, full N samples). Outcome is logged and recorded in sync_metrics.json.
        # Video is NOT cut (stream copy cannot drop the extra last frame: it is a reference frame of displayed B-frames).
        import soundfile as _sf
        _why = None
        N_own, _exact, _why = _chunk_owned_samples(out_dir, N, sr)
        if _exact:
            try:
                _w = final[:N_own]
                save_wav_stereo(tmp_wav, _w, sr)
                _inf = _sf.info(str(tmp_wav))
                if _inf.frames != N_own or _inf.samplerate != sr:
                    raise RuntimeError(f"dub.wav validation failed: frames {_inf.frames} != {N_own}")
                run_ffmpeg(["-i", str(out_dir / "src.mp4"), "-i", str(tmp_wav), "-map", "0:v:0", "-map", "1:a:0",
                            "-c:v", "copy", "-c:a", "aac", str(tmp_mp4)])
                _ad = _audio_track_dur(tmp_mp4)
                if not media_ok(tmp_mp4, need_video=True, need_audio=True, min_dur=max(0.05, 0.9 * N_own / sr - 1.0)):
                    raise RuntimeError("dub mp4 validation failed (ffprobe)")
                if _ad is None or not math.isfinite(_ad) or abs(_ad - N_own / sr) > 0.003:
                    raise RuntimeError(f"audio track {_ad} s != owned {N_own / sr:.4f} s")
            except Exception as _e:
                _why = f"{type(_e).__name__}: {_e}"
                _exact = False
        if not _exact:
            N_own = N
            log.warning("mix: LEGACY mux path taken for this chunk (exact_own=false): %s", _why)
            save_wav_stereo(tmp_wav, final, sr)
            _inf = _sf.info(str(tmp_wav))
            if _inf.frames != N or _inf.samplerate != sr:
                raise RuntimeError(f"dub.wav validation failed: frames {_inf.frames} != {N}")
            run_ffmpeg(["-i", str(out_dir / "src.mp4"), "-i", str(tmp_wav), "-map", "0:v:0", "-map", "1:a:0",
                        "-c:v", "copy", "-c:a", "aac", "-shortest", str(tmp_mp4)])
        try:
            atomic_write_json(out_dir / "sync_metrics.json", {"exact_own": bool(_exact), "reason": _why, "n_samples": int(N),
                              "owned_samples": int(N_own), "audio_track_s": _audio_track_dur(tmp_mp4)})
        except Exception as _e:
            log.warning("sync_metrics.json not written: %s", _e)
        if not media_ok(tmp_mp4, need_video=True, need_audio=True, min_dur=max(0.05, 0.9 * N_own / sr - 1.0)):
            raise RuntimeError("dub mp4 validation failed (ffprobe)")
        os.replace(tmp_wav, out_wav)
        os.replace(tmp_mp4, out_mp4)
    finally:
        for _p in (tmp_wav, tmp_mp4):
            try:
                os.remove(_p)
            except OSError:
                pass
    log.info("Stage mix: done -> %s", out_mp4)
    return out_mp4

# ---------------------------------------------------------------------------
# Resynth
# ---------------------------------------------------------------------------
def do_resynth(args, out_dir: Path):
    seg_path = out_dir / "segments.json"
    if not seg_path.exists():
        raise RuntimeError("segments.json not found; run full pipeline first")
    with open(seg_path, encoding="utf-8") as f:
        segments = json.load(f)
    seg = next((s for s in segments if s["id"] == args.resynth), None)
    if seg is None:
        raise RuntimeError(f"segment id {args.resynth} not found")

    # edit history is owned by the server (edit pushes, undo pops); appending here made undo a no-op
    seg["tgt_text"] = args.text
    seg["edited"] = True
    seg.pop("sent_pieces", None)       # the edit is whole-slot audio; drop stale placement metadata
    seg.pop("slack_added", None)

    keys = load_keys()
    session = thread_session()
    tts_key = keys["tts"]
    tts_dir = out_dir / "tts"
    tts_dir.mkdir(exist_ok=True)
    raw_path = tts_dir / f"{args.resynth}_raw.wav"
    out_path = tts_dir / f"{args.resynth}.wav"
    voice_require_speakers([seg])
    try:
        voice = pick_voice(args.lang, seg["gender"], seg["speaker"])
        _fault("tts", args.resynth)
        wav_bytes = bodhan_tts(session, tts_key, args.text, args.lang, voice, style=_expr_style(seg))
        write_raw_tts(raw_path, wav_bytes, tts_fingerprint(args.text, voice, args.lang, _expr_style(seg)))
    except Exception as e:
        log.error("Resynth TTS error: %s", e)
        raise
    # refit
    fit_src, tts_dur = _pretrim_wav(raw_path, tts_dir / f"{args.resynth}_trim.wav")
    slot_dur = max(seg["end"] - seg["start"], 1e-6)
    ratio = tts_dur / slot_dur
    tempo = min(ratio, 1.25) if ratio > 1.0 else 1.0
    run_ffmpeg([
        "-i", str(fit_src),
        "-filter:a", f"atempo={tempo:.6f}",
        "-ac", "1", "-ar", "24000",
        str(out_path),
    ])
    if fit_src is not raw_path:
        try:
            os.remove(fit_src)
        except OSError:
            pass
    fitted_dur = librosa.get_duration(path=str(out_path))
    nxt = next((s for s in segments if s["id"] == seg["id"] + 1), None)
    allowed = slot_dur
    if nxt is not None:
        allowed = max(slot_dur, nxt["start"] - seg["start"])
    if fitted_dur > allowed + 1e-3:
        seg["overflow"] = True
        seg["overflow_s"] = fitted_dur - allowed
        run_ffmpeg([
            "-i", str(out_path), "-t", f"{allowed}",
            "-af", f"afade=t=out:st={max(0,allowed-0.05)}:d=0.05",
            "-ac", "1", "-ar", "24000", str(out_path),
        ])
    else:
        seg["overflow"] = False
        seg["overflow_s"] = 0.0
    seg["ratio"] = ratio
    seg["tempo"] = tempo
    seg["voice"] = voice
    # a successful re-voicing clears any earlier fallback flag on this segment
    if seg.get("fallback"):
        for k in ("fallback", "fallback_reason", "tts_error", "mt_error", "asr_error"):
            seg.pop(k, None)
        seg["error"] = None

    atomic_write_json(seg_path, segments, indent=2)

    # redo mix (bed cached)
    out_mp4 = stage_mix(args, out_dir, segments)
    print(out_mp4)
    return out_mp4

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def load_timings(out_dir: Path) -> Dict[str, float]:
    p = out_dir / "timings.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_timing(out_dir: Path, stage: str, secs: float):
    t = load_timings(out_dir)
    t[stage] = round(t.get(stage, 0.0) + secs, 3)
    atomic_write_json(out_dir / "timings.json", t, indent=2)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(args, out_dir: Path, segments: List[Dict[str, Any]], out_mp4: Path):
    n = len(segments)
    spk_g: Dict[int, str] = {}
    for s in segments:
        spk_g[s["speaker"]] = s["gender"]
    ratios = [s.get("ratio", 1.0) for s in segments if not s["error"]]
    mean_r = float(np.mean(ratios)) if ratios else 0.0
    max_r = float(np.max(ratios)) if ratios else 0.0
    ovf = sum(1 for s in segments if s.get("overflow"))
    timings = load_timings(out_dir)
    log.info("=== SUMMARY ===")
    log.info("segments: %d", n)
    for spk, g in sorted(spk_g.items()):
        log.info("  speaker %d: %s", spk, g)
    log.info("fit ratio mean=%.3f max=%.3f", mean_r, max_r)
    log.info("overflow segments: %d", ovf)
    for stage in ["fetch", "audio", "vad", "slots", "speakers", "asr", "mt", "tts", "separate", "mix"]:
        if stage in timings:
            log.info("  %-9s: %.2fs", stage, timings[stage])
    log.info("output: %s", out_mp4)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _early_chunk(out_dir: Path) -> bool:
    """Latency-critical first chunks (index < REFIT_MIN_CHUNK, default 2) skip the serial Gemini refit + re-synthesis;
    an overflowing segment is then handled by the tempo cap + truncation-with-fade already in stage_tts."""
    import re
    try:
        m = re.fullmatch(r"c(\d+)", Path(out_dir).name)
        return bool(m) and int(m.group(1)) < int(os.environ.get("REFIT_MIN_CHUNK", "0"))
    except Exception:
        return False


def chunk_lookahead_gate(out_dir: Path):
    """Playback needs chunks in order, but the server starts every chunk of every language at once, so chunk n+1..n+3
    steal TTS/MT/CPU from chunk n and ALL chunks finish late. Keep a window of CHUNK_LOOKAHEAD chunks per language:
    chunk k starts only when chunk k-L of the same language is done/error (or after CHUNK_GATE_MAX_WAIT=360 s (bounds a stale running state)). L=0 disables."""
    import re
    try:
        L = int(os.environ.get("CHUNK_LOOKAHEAD", "3"))
        m = re.fullmatch(r"c(\d+)", out_dir.name)
        if L <= 0 or not m or int(m.group(1)) < L:
            return
        k = int(m.group(1))
        prev = out_dir.parent / f"c{k - L:04d}" / "status.json"
        deadline = time.time() + float(os.environ.get("CHUNK_GATE_MAX_WAIT", "360"))
        t0 = time.time()
        while time.time() < deadline:
            try:
                st = json.load(open(prev, encoding="utf-8")).get("state")
            except Exception:
                st = None
            if st in ("done", "error", "failed"):
                break
            time.sleep(1.0)
        log.info("chunk gate: waited %.1fs for %s (state=%s)", time.time() - t0, prev.parent.name, st)
        save_timing(out_dir, "gate_wait", time.time() - t0)
    except Exception as e:  # never block the pipeline on a gate bug
        log.warning("chunk gate failed (%s); continuing", e)


def _prep_stages(args, out_dir: Path):
    """language-independent front half: fetch -> audio -> vad -> slots -> speakers -> asr (source-language transcript)."""
    t0 = time.time()
    meta = stage_fetch(args, out_dir)
    save_timing(out_dir, "fetch", time.time() - t0)

    t0 = time.time(); stage_audio(out_dir); save_timing(out_dir, "audio", time.time() - t0)
    t0 = time.time(); vad = stage_vad(out_dir); save_timing(out_dir, "vad", time.time() - t0)
    t0 = time.time(); slots = stage_slots(out_dir, vad); save_timing(out_dir, "slots", time.time() - t0)
    t0 = time.time(); _ = stage_speakers(out_dir, slots); save_timing(out_dir, "speakers", time.time() - t0)
    t0 = time.time(); segments = stage_asr(args, out_dir, slots); save_timing(out_dir, "asr", time.time() - t0)
    return segments


_PREP_FILES = ["meta.json", "src.mp4", "src16k.wav", "src44k.wav", "vad.json", "slots.json", "speakers.json",
               "lid.json", "segments.json", "crops", ".lid"]


def _zero_slots(d: Path) -> bool:
    """True when <d>/slots.json parses to an EMPTY list (VAD found no speech): an empty crops dir is then complete, not partial."""
    try:
        v = json.load(open(Path(d) / "slots.json", encoding="utf-8"))
        v = v.get("slots", v) if isinstance(v, dict) else v
        return isinstance(v, list) and len(v) == 0
    except Exception:
        return False


def _prep_file_ok(path: Path) -> bool:
    """A prep artefact is usable only if it is complete: wav decodes, mp4 probes, json parses, dirs non-empty."""
    try:
        if not path.exists():
            return False
        if path.is_dir():
            # a chunk with NO speech has zero slots and therefore an empty crops dir; that is a complete, valid prep (the dub keeps the original audio)
            return path.name != "crops" or any(path.iterdir()) or _zero_slots(path.parent)
        suf = path.suffix.lower()
        if suf == ".wav":
            return _wav_ok(path)
        if suf == ".mp4":
            return media_ok(path, need_video=True, need_audio=True, min_dur=0.05)
        if suf == ".json":
            json.load(open(path, encoding="utf-8"))
            return True
        return path.stat().st_size >= 0
    except Exception:
        return False


def _atomic_copy(src: Path, dst: Path):
    """copy file/dir to dst via a temp name + rename so a crash never leaves a partial file at the final path"""
    import shutil
    tmp = dst.with_name(dst.name + ".tmpcopy")
    if tmp.exists():
        shutil.rmtree(tmp) if tmp.is_dir() else tmp.unlink()
    if src.is_dir():
        shutil.copytree(src, tmp)
        if dst.exists():
            shutil.rmtree(dst)
    else:
        shutil.copy2(src, tmp)
    os.replace(tmp, dst)


# ---------------------------------------------------------------------------
# NEW-JOB chunk generations (speaker-split slots). Design: the split decision is made ONCE, when a chunk's shared
# prep (<job>/_prep/cNNNN) is first created under its flock. Everything that existed before (no slot_gen.json) is LEGACY and
# runs the old prep_shared body unchanged (no checks, no markers). A new generation = vad.json + slots.json + speakers.json
# + slot_gen.json (marker, written LAST = the commit record). The marker fingerprints the inputs (SPEAKER_MAP content,
# CHUNK_START, thresholds, VAD content) and the produced slots/speakers. Language dirs take the WHOLE generation from _prep
# (replacing any local copy that is not byte-identical) and verify it; every failure here is a hard error.
# ---------------------------------------------------------------------------
GEN_FILE = "slot_gen.json"
_GEN_CORE = ["vad.json", "slots.json", "speakers.json"]
_SLOT_DEP = ["slots.json", "speakers.json", "segments.json", "segments.failed.json", "slot_gen.json", "fit.json",
             "lipsync.json", "crops", "tts"]
_NEWGEN_PENDING = ".newgen_creating"


def _sha_file(p) -> str:
    import hashlib
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:20]
    except OSError:
        return "missing"


def _gen_inputs(vad_path) -> Dict[str, Any]:
    return {"smap": _sha_file(os.environ.get("SPEAKER_MAP", "")), "chunk_start": str(os.environ.get("CHUNK_START", "0")),
            "min_spk": SPLIT_MIN_SPK_S, "min_piece": SPLIT_MIN_PIECE_S, "vad": _sha_file(vad_path)}


def _gen_marker(d: Path) -> Dict[str, Any]:
    return {"gen": 1, "inputs": _gen_inputs(d / "vad.json"), "slots": _sha_file(d / "slots.json"),
            "speakers": _sha_file(d / "speakers.json")}


def _gen_read(d: Path):
    p = d / GEN_FILE
    if not p.exists():
        return None
    m = json.load(open(p, encoding="utf-8"))         # corrupt marker -> exception (hard)
    if not isinstance(m, dict) or m.get("gen") != 1:
        raise RuntimeError(f"{p}: unsupported slot generation marker")
    return m


def _gen_verify(d: Path) -> Dict[str, Any]:
    """The marker in d must match the bytes of d's vad/slots/speakers. Raises RuntimeError on any mismatch."""
    m = _gen_read(d)
    if m is None:
        raise RuntimeError(f"{d}: slot generation marker missing")
    for k, f in (("slots", "slots.json"), ("speakers", "speakers.json")):
        if m.get(k) != _sha_file(d / f):
            raise RuntimeError(f"{d}: {f} does not match its generation fingerprint (marker {m.get(k)} vs {_sha_file(d / f)})")
    if m["inputs"].get("vad") != _sha_file(d / "vad.json"):
        raise RuntimeError(f"{d}: vad.json does not match its generation fingerprint")
    return m


def _gen_inputs_changed(m: Dict[str, Any]) -> bool:
    cur = _gen_inputs(Path("/nonexistent"))
    return any(m["inputs"].get(k) != cur[k] for k in ("smap", "chunk_start", "min_spk", "min_piece"))


def _purge_slot_dep(d: Path):
    import shutil
    for n in _SLOT_DEP:
        p = d / n
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists() or p.is_symlink():
            p.unlink()
    for p in list(d.glob("dub*")) + list(d.glob(".dubtmp*")):
        try:
            p.unlink()
        except OSError:
            pass


def _chunk_mode(job_dir: Path, out_dir: Path, sh: Path) -> str:
    """'new' | 'legacy', decided under the chunk's flock. Anything that predates the marker is legacy."""
    if (sh / GEN_FILE).exists() or (sh / _NEWGEN_PENDING).exists():
        return "new"
    if not SPLIT_SPK:
        return "legacy"
    sm = os.environ.get("SPEAKER_MAP")
    if not (sm and os.path.exists(sm)):
        return "legacy"
    if (sh / "PREP_DONE").exists():
        return "legacy"
    for d in job_dir.glob(f"*/{out_dir.name}"):
        if any((d / n).exists() for n in ("slots.json", "segments.json", "speakers.json")):
            return "legacy"
    return "new"


def _publish_files(src: Path, dst: Path, names):
    for n in names:
        if (src / n).exists():
            if not _prep_file_ok(src / n):
                raise RuntimeError(f"refusing to publish partial prep file {n}")
            _atomic_copy(src / n, dst / n)


def _prep_shared_new(args, out_dir: Path, job_dir: Path, sh: Path):
    import shutil
    done = sh / "PREP_DONE"
    m = _gen_read(sh)
    if m is not None and _gen_inputs_changed(m):
        log.error("slot generation inputs changed; re-plan required (%s): SPEAKER_MAP/CHUNK_START/thresholds differ from the "
                  "fingerprint in %s. Nothing was modified.", sh, GEN_FILE)
        raise RuntimeError("slot generation inputs changed; re-plan required")
    if m is not None:
        # fence: core files must not be newer than the commit record (an old-binary writer after the commit)
        mt = (sh / GEN_FILE).stat().st_mtime
        for n in _GEN_CORE:
            if (sh / n).stat().st_mtime > mt + 1.0:
                raise RuntimeError(f"{sh / n} was modified after its slot generation was committed")
    if m is None:
        # ---- create the generation (first language to hold the flock) ----
        _purge_slot_dep(out_dir)
        (sh / _NEWGEN_PENDING).touch()
        _NEWGEN["on"] = True
        try:
            segments = _prep_stages(args, out_dir)
        except BaseException:
            _purge_slot_dep(out_dir)         # leave nothing that a retry could mistake for a legacy chunk
            raise
        finally:
            _NEWGEN["on"] = False
        marker = _gen_marker(out_dir)
        atomic_write_json(out_dir / GEN_FILE, marker)        # failure -> exception -> chunk fails (no swallow)
        _publish_files(out_dir, sh, _GEN_CORE)
        atomic_write_json(sh / GEN_FILE, marker)             # COMMIT RECORD, written last
        _gen_verify(sh)
        (sh / _NEWGEN_PENDING).unlink()
        log.info("Prep shared: new slot generation committed (%d slots) at %s", len(segments), sh)
        if not any(sg.get("fallback_reason") == "asr" for sg in segments):
            _publish_files(out_dir, sh, [n for n in _PREP_FILES if n not in _GEN_CORE])
            done.touch()
        return segments
    # ---- consume an existing generation ----
    m = _gen_verify(sh)
    lm = None
    try:
        lm = _gen_read(out_dir)
    except Exception:
        lm = None
    same = lm == m and all(_sha_file(out_dir / f) == _sha_file(sh / f) for f in _GEN_CORE)
    if not same:
        log.warning("Prep shared: replacing local slot-dependent artifacts of %s with the shared generation", out_dir)
        _purge_slot_dep(out_dir)
    for n in _GEN_CORE:
        if not same or not (out_dir / n).exists():
            _atomic_copy(sh / n, out_dir / n)
    _atomic_copy(sh / GEN_FILE, out_dir / GEN_FILE)
    _gen_verify(out_dir)
    if done.exists():
        for n in _PREP_FILES:
            if n in _GEN_CORE or n == GEN_FILE:
                continue
            src, dst = sh / n, out_dir / n
            if src.exists() and not (dst.exists() and _prep_file_ok(dst)):
                if not _prep_file_ok(src):
                    raise RuntimeError(f"shared prep file {n} is stale/partial")
                _atomic_copy(src, dst)
        segments = _reset_downstream(json.load(open(out_dir / "segments.json", encoding="utf-8")))
    else:
        segments = _prep_stages(args, out_dir)          # creator had ASR failures: retry on the copied generation
        if not any(sg.get("fallback_reason") == "asr" for sg in segments):
            _publish_files(out_dir, sh, [n for n in _PREP_FILES if n not in _GEN_CORE])
            done.touch()
    sl = json.load(open(out_dir / "slots.json", encoding="utf-8"))
    if [x["id"] for x in sl] != [x["id"] for x in segments]:
        raise RuntimeError(f"{out_dir}: segments ids differ from the slot generation")
    return segments


def prep_shared(args, out_dir: Path):
    """Compute the language-independent front half ONCE per (job, chunk) and share it across target languages.
    Layout <job>/<lang>/cNNNN: the first language to take the flock computes and publishes to <job>/_prep/cNNNN,
    the others copy it (PREP_DONE is written last). Any failure falls back to computing locally."""
    import fcntl, shutil
    job_dir = out_dir.parent.parent
    if os.environ.get("PREP_SHARE", "1") != "1" or not (job_dir / "plan.json").exists() or not out_dir.name.startswith("c"):
        return _prep_stages(args, out_dir)
    sh = job_dir / "_prep" / out_dir.name
    sh.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(sh / ".lock", "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        if _chunk_mode(job_dir, out_dir, sh) == "new":
            return _prep_shared_new(args, out_dir, job_dir, sh)          # new-job generation (hard errors)
        done = sh / "PREP_DONE"                                           # ---- LEGACY path: unchanged below ----
        if done.exists():
            try:
                for n in _PREP_FILES:
                    src, dst = sh / n, out_dir / n
                    if not src.exists():
                        continue
                    if dst.exists() and _prep_file_ok(dst):
                        continue            # complete local copy
                    if not _prep_file_ok(src):
                        raise RuntimeError(f"shared prep file {n} is stale/partial")
                    _atomic_copy(src, dst)
                with open(out_dir / "segments.json", encoding="utf-8") as f:
                    segments = _reset_downstream(json.load(f))
                log.info("Prep shared: copied from %s in %.1fs", sh, time.time() - t0)
                save_timing(out_dir, "prep_shared_copy", time.time() - t0)
                return segments
            except Exception as e:
                log.warning("shared prep copy failed (%s); recomputing locally and republishing", e)
                try:
                    done.unlink()
                except OSError:
                    pass
                for n in _PREP_FILES:
                    if (out_dir / n).exists() and not _prep_file_ok(out_dir / n):
                        shutil.rmtree(out_dir / n) if (out_dir / n).is_dir() else (out_dir / n).unlink()
                segments = _prep_stages(args, out_dir)
                if any(sg.get("fallback_reason") == "asr" for sg in segments):
                    return segments
                try:
                    for n in _PREP_FILES:
                        if (out_dir / n).exists() and _prep_file_ok(out_dir / n):
                            _atomic_copy(out_dir / n, sh / n)
                    done.touch()
                except Exception as e2:
                    log.warning("could not republish shared prep (%s)", e2)
                return segments
        segments = _prep_stages(args, out_dir)
        if any(sg.get("fallback_reason") == "asr" for sg in segments):
            log.warning("Prep shared: NOT publishing (some ASR slots failed; each language retries them)")
            return segments
        try:
            for n in _PREP_FILES:
                src, dst = out_dir / n, sh / n
                if not src.exists():
                    continue
                if not _prep_file_ok(src):
                    raise RuntimeError(f"refusing to publish partial prep file {n}")
                _atomic_copy(src, dst)
            done.touch()
            log.info("Prep shared: published to %s", sh)
        except Exception as e:
            log.warning("could not publish shared prep (%s)", e)
        return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--lang", required=True, choices=sorted(LANGS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--dur", type=float, default=90.0)
    parser.add_argument("--resynth", type=int, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    voice_preflight(args.lang, out_dir)           # outside every segment-level handler: a bad plan fails the chunk

    # stash duration for slot force-cut
    global _args_dur
    _args_dur = args.dur

    if args.resynth is not None:
        if not args.text:
            parser.error("--resynth requires --text")
        do_resynth(args, out_dir)
        print("ALL DONE")
        return

    # Full pipeline
    chunk_lookahead_gate(out_dir)
    segments = prep_shared(args, out_dir)
    t0 = time.time(); segments = stage_mt(args, out_dir, segments); save_timing(out_dir, "mt", time.time() - t0)
    from sunwai.pipeline.fitting.fit_stage import stage_fit
    args.pick_voice = pick_voice                       # stage_fit budgets per voice
    t0 = time.time(); segments = stage_fit(args, out_dir, segments); save_timing(out_dir, "fit", time.time() - t0)
    t0 = time.time(); segments = stage_numbers(args, out_dir, segments); save_timing(out_dir, "numbers", time.time() - t0)
    t0 = time.time(); segments = stage_tts(args, out_dir, segments); save_timing(out_dir, "tts", time.time() - t0)
    t0 = time.time(); stage_separate(out_dir); save_timing(out_dir, "separate", time.time() - t0)
    t0 = time.time(); out_mp4 = stage_mix(args, out_dir, segments); save_timing(out_dir, "mix", time.time() - t0)

    print_summary(args, out_dir, segments, out_mp4)
    print("ALL DONE")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted")
        sys.exit(130)
    except Exception as e:
        log.error("Fatal error: %s", e)
        log.debug(traceback.format_exc())
        sys.exit(1)
