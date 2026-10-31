"""ASR stage: transcribe slots and deduplicate boundary words."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List


log = logging.getLogger(__name__)

from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.stage import StagePolicy
from sunwai.pipeline.models import from_records
from sunwai.pipeline.stages.base import Segments, StageBase

from sunwai.clients.service_calls import bodhan_asr, load_keys
from sunwai.media.audio import crop_to_seconds, load_wav_mono, save_wav_mono
from sunwai.pipeline.stages.slots import _cut_points
from sunwai.pipeline.stages.speakers import detect_source_language
from sunwai.pipeline.support.concurrency import _fault, thread_session
from sunwai.pipeline.support.guards import enforce_fallback_budget, mark_fallback
from sunwai.pipeline.support.storage import atomic_write_json


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


class AsrStage(StageBase[list[dict[str, Any]], Segments]):
    """Transcribe slots and deduplicate boundary words."""

    name = "asr"
    policy = StagePolicy(timing_key="asr")

    def run(self, value: list[dict[str, Any]], ctx: ChunkContext) -> Segments:
        """Transcribe slots and deduplicate boundary words."""
        return from_records(stage_asr(ctx.legacy_args(), ctx.work_dir, value))
