"""MT stage: translation with source-span routing and script recovery."""

from __future__ import annotations

import logging
import os
import threading as _threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import requests

log = logging.getLogger(__name__)

from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.stage import StagePolicy
from sunwai.pipeline.models import from_records, to_records
from sunwai.pipeline.stages.base import Segments, StageBase

from sunwai.clients.service_endpoints import IT2_BASE, IT2_TARGETS, MT_BASES
from sunwai.clients.service_calls import _budget_timeout, _stream_post, bodhan_mt, load_keys
from sunwai.pipeline.stages.speakers import _source_lang
from sunwai.pipeline.support.concurrency import _fault, inflight, thread_session
from sunwai.pipeline.support.env import _env_clamped
from sunwai.pipeline.support.guards import enforce_fallback_budget, mark_fallback
from sunwai.pipeline.support.storage import atomic_write_json


def routed_mt(session, key, text, lang, src):
    """Translate one string, routing English to the LLM, IndicTrans2 targets to the mirror and the rest to Bodhan MT."""
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


MT_PIECE_WORDS = _env_clamped("MT_PIECE_WORDS", 12, 4, 40, int)


MT_RECOVER_BUDGET_S = _env_clamped("MT_RECOVER_BUDGET_S", 45.0, 5.0, 60.0)


MT_RECOVER_MAX_PIECES = _env_clamped("MT_RECOVER_MAX_PIECES", 6, 2, 12, int)


MT_RECOVER_CALL_S = _env_clamped("MT_RECOVER_CALL_S", 15.0, 1.0, 60.0)       # per-call POST timeout (limiter wait excluded)


MT_RECOVER_CAP_S = _env_clamped("MT_RECOVER_CAP_S", 120.0, 10.0, 180.0)      # hard WALL-CLOCK ceiling per segment (queue wait included)


MT_RECOVER_CHUNK_S = _env_clamped("MT_RECOVER_CHUNK_S", 180.0, 10.0, 600.0)   # total recovery work per chunk process (summed over segments)


_MT_SPENT = [0.0]          # chunk-wide pool already COMMITTED (spent + reserved by in-flight recoveries); guarded by _MT_SPENT_LOCK


_MT_SPENT_LOCK = _threading.Lock()


MT_RECOVER_EXTEND_S = _env_clamped("MT_RECOVER_EXTEND_S", 20.0, 0.0, 60.0)   # budget added per observed TIMEOUT (not per script error)


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
    """Translate every non-edited segment into ``args.lang`` and record failures per segment.

    Args:
        args: Chunk arguments (target language).
        out_dir: Chunk directory.
        segments: Segment records with ``src_text``.

    Returns:
        The same records with ``tgt_text`` filled in.
    """
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


class MtStage(StageBase[Segments, Segments]):
    """Translate segments into the target language with source-span routing."""

    name = "mt"
    policy = StagePolicy(timing_key="mt")

    def run(self, value: Segments, ctx: ChunkContext) -> Segments:
        """Translate segments into the target language with source-span routing."""
        return from_records(stage_mt(ctx.legacy_args(), ctx.work_dir, to_records(value)))
