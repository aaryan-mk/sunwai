"""Preview: synthesise a CANDIDATE revision of a line in a new delivery. Never touches the live audio."""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sunwai.control.expression import engine_bridge
from sunwai.control.expression.errors import ExprError, enabled
from sunwai.control.expression.fitting import (
    Take,
    TempoFn,
    TrimFn,
    TtsFn,
    ffmpeg_tempo,
    pad_to,
    synth_fit,
    wav_duration,
)
from sunwai.control.expression.fsutil import FileLock, read_json, sha256_file, write_json_atomic
from sunwai.control.expression.planner import plan_expression
from sunwai.control.expression.revisions import (
    Segment,
    expr_dir,
    find_segment,
    head_revision,
    load_segments,
    update_operation,
)
from sunwai.control.expression.voices import resolve_voice

VoiceFn = Callable[[Path, str, Segment], str]


class CostMeter:
    """Wraps a synthesiser and charges every attempt, successful or not."""

    def __init__(self, tts: TtsFn) -> None:
        """Wrap ``tts`` with zeroed counters."""
        self._tts = tts
        self.cost: dict[str, Any] = {"tts_calls": 0, "tts_chars": 0, "attempts": []}

    def __call__(self, text: str, lang: str, voice: str, style: str | None) -> bytes:
        """Record the attempt, then synthesise."""
        self.cost["tts_calls"] += 1
        self.cost["tts_chars"] += len(text)
        self.cost["attempts"].append({"chars": len(text), "style": style})
        return self._tts(text, lang, voice, style)

    def charged(self) -> dict[str, int]:
        """Return the totals charged by this call."""
        return {"tts_calls": self.cost["tts_calls"], "tts_chars": self.cost["tts_chars"]}


def slot_limits(segments: list[Segment], seg: Segment) -> tuple[float, float]:
    """Return ``(slot_s, allowed_s)``: the line's own slot and the longest take that may run into the gap before the next line."""
    slot = max(seg["end"] - seg["start"], 1e-6)
    following = next((s for s in segments if s["id"] == seg["id"] + 1), None) if isinstance(seg["id"], int) else None
    allowed = max(slot, following["start"] - seg["start"]) if following else slot
    return slot, allowed


def _idempotency_path(out_dir: Path, seg_id: str | int, key: str) -> Path:
    directory = expr_dir(out_dir) / "idem"
    directory.mkdir(exist_ok=True)
    return directory / (hashlib.sha1(f"{seg_id}|{key}".encode()).hexdigest() + ".json")


class CandidateSynthesizer:
    """Builds one candidate record: plan, synthesise, fit, repair once, then classify the outcome."""

    def __init__(
        self,
        out_dir: Path,
        lang: str,
        seg: Segment,
        tts: TtsFn,
        trim: TrimFn,
        tempo_fn: TempoFn,
        slot_s: float,
        allowed_s: float,
    ) -> None:
        """Bind the line and the injected audio functions."""
        self.out_dir, self.lang, self.seg = out_dir, lang, seg
        self.tts, self.trim, self.tempo_fn = tts, trim, tempo_fn
        self.slot_s, self.allowed_s = slot_s, allowed_s

    def _take(self, plan: dict[str, Any], voice: str, tag: str, cand_dir: Path) -> Take:
        return synth_fit(
            plan["spoken_text"],
            self.lang,
            plan["style"],
            voice,
            cand_dir,
            tag,
            tts=self.tts,
            trim=self.trim,
            tempo_fn=self.tempo_fn,
            allowed_s=self.allowed_s,
            slot_s=self.slot_s,
        )

    def synthesise(
        self,
        rec: dict[str, Any],
        plan: dict[str, Any],
        intent: str,
        wording_patch: str | None,
        voice: str,
        cand_dir: Path,
    ) -> None:
        """Fill ``rec`` with the outcome of synthesis: one take, plus ONE duration-informed repair when it overruns."""
        cid = rec["candidate_id"]
        take = self._take(plan, voice, cid, cand_dir)
        if take.overruns:
            repair = plan_expression(self.seg["tgt_text"], self.lang, intent, compact=True, wording_patch=wording_patch)
            take = self._take(repair, voice, cid + "r", cand_dir)
            rec.update(repair_used=True, spoken_text=repair["spoken_text"], style=repair["style"], plan_repair=repair)
        if take.overruns:
            reason = f"overrun: {take.fitted_s:.2f}s > {self.allowed_s:.2f}s after repair"
            rec.update(status="rejected", reason=reason, fitted_s=take.fitted_s, tempo_factor=take.tempo)
            update_operation(self.out_dir, cid, state="error", stage="rejected", error=reason)
            return
        if take.fitted_s < self.slot_s:
            pad_to(take.path, self.slot_s)
        rec.update(
            audio_path=str(take.path),
            audio_hash=sha256_file(take.path),
            raw_s=take.raw_s,
            fitted_s=wav_duration(take.path),
            tempo_factor=take.tempo,
            status="ready",
        )

    def settle_against_head(self, rec: dict[str, Any], lang: str, seg_id: str | int, base_revision: int) -> None:
        """Mark a ready candidate superseded when an undo/edit moved the head during generation."""
        cid = rec["candidate_id"]
        if head_revision(self.out_dir, lang, seg_id)["revision"] != base_revision:
            rec["status"] = "superseded"
            update_operation(self.out_dir, cid, state="cancelled", stage="superseded")
        else:
            update_operation(self.out_dir, cid, state="running", stage="audio ready", audio_ready=True)


def do_expression_resynth(
    out_dir: str | Path,
    lang: str,
    seg_id: str | int,
    intent: str | None,
    expected_revision: int,
    idempotency_key: str,
    *,
    job_id: str | None = None,
    action_id: str | None = None,
    wording_patch: str | None = None,
    tts: TtsFn | None = None,
    voice_fn: VoiceFn | None = None,
    trim: TrimFn | None = None,
    tempo_fn: TempoFn | None = None,
) -> dict[str, Any]:
    """Synthesise a CANDIDATE revision; it never touches the active audio.

    A repeated ``idempotency_key`` replays the stored result at no cost. Every TTS attempt is charged.

    Args:
        out_dir: Chunk directory.
        lang: Language code.
        seg_id: Segment id.
        intent: Delivery intent (see :data:`~sunwai.control.expression.planner.INTENT_STYLE`).
        expected_revision: Head revision the caller saw (compare-and-swap).
        idempotency_key: Caller-chosen key; required.
        job_id: Job id recorded on the candidate.
        action_id: Action id recorded on the candidate.
        wording_patch: Optional validated replacement wording.
        tts: Synthesiser override (tests).
        voice_fn: Voice resolver override (tests).
        trim: Trimmer override (tests).
        tempo_fn: Time-stretch override (tests).

    Returns:
        The candidate record, with ``status`` of ``ready``, ``rejected``, ``superseded`` or ``error``.

    Raises:
        ExprError: 503 when disabled, 422 for bad input, 409 for a stale revision.
    """
    if not enabled():
        raise ExprError(503, "expression disabled (EXPRESSION_ENABLED=0)")
    if not idempotency_key:
        raise ExprError(422, "idempotency_key required")
    meter = CostMeter(tts or engine_bridge.default_tts)
    out_dir = Path(out_dir)
    stored = _idempotency_path(out_dir, seg_id, idempotency_key)
    with FileLock(str(stored) + ".lock"):  # duplicate submissions serialise, the 2nd returns the 1st result
        previous = read_json(stored)
        if previous:
            previous["duplicate"] = True
            previous["cost_this_call"] = {"tts_calls": 0, "tts_chars": 0}  # a replay charges nothing
            return previous  # type: ignore[no-any-return]
        head = head_revision(out_dir, lang, seg_id)
        if head["revision"] != int(expected_revision):
            raise ExprError(409, f"stale revision: expected {expected_revision}, head {head['revision']}")
        segments = load_segments(out_dir)
        seg = find_segment(segments, seg_id)
        if not (seg.get("tgt_text") or "").strip():
            raise ExprError(422, "segment has no spoken text")
        rec = _build_candidate(
            out_dir,
            lang,
            seg,
            segments,
            head["revision"],
            intent or "",
            meter,
            job_id=job_id,
            action_id=action_id,
            wording_patch=wording_patch,
            voice_fn=voice_fn,
            trim=trim or engine_bridge.default_trim,
            tempo_fn=tempo_fn or ffmpeg_tempo,
        )
        rec["cost_this_call"] = meter.charged()
        write_json_atomic(expr_dir(out_dir) / "cands" / f"{rec['candidate_id']}.json", rec)
        write_json_atomic(stored, rec)
        return rec


def _build_candidate(
    out_dir: Path,
    lang: str,
    seg: Segment,
    segments: list[Segment],
    base_revision: int,
    intent: str,
    meter: CostMeter,
    *,
    job_id: str | None,
    action_id: str | None,
    wording_patch: str | None,
    voice_fn: VoiceFn | None,
    trim: TrimFn,
    tempo_fn: TempoFn,
) -> dict[str, Any]:
    cid = "c" + uuid.uuid4().hex[:10]
    update_operation(out_dir, cid, action_id=action_id, state="running", stage="synthesizing", revision=base_revision)
    plan = plan_expression(seg["tgt_text"], lang, intent, wording_patch=wording_patch)
    slot, allowed = slot_limits(segments, seg)
    cand_dir = expr_dir(out_dir) / "cands"
    cand_dir.mkdir(exist_ok=True)
    voice = (
        voice_fn(out_dir, lang, seg) if voice_fn else resolve_voice(out_dir, lang, seg)
    )  # the voice plan is respected, never changed
    rec: dict[str, Any] = {
        "candidate_id": cid,
        "operation_id": cid,
        "action_id": action_id,
        "job_id": job_id,
        "lang": lang,
        "segment_id": seg["id"],
        "base_revision": base_revision,
        "intent": plan["intent"],
        "style": plan["style"],
        "spoken_text": plan["spoken_text"],
        "plan": plan,
        "voice": voice,
        "slot_s": slot,
        "allowed_s": allowed,
        "repair_used": False,
        "status": "running",
        "ts": time.time(),
        "cost": meter.cost,
    }
    builder = CandidateSynthesizer(out_dir, lang, seg, meter, trim, tempo_fn, slot, allowed)
    try:
        builder.synthesise(rec, plan, intent, wording_patch, voice, cand_dir)
        if rec["status"] == "ready":
            builder.settle_against_head(rec, lang, seg["id"], base_revision)
    except ExprError:
        raise
    except Exception as exc:
        rec.update(status="error", reason=str(exc))
        update_operation(out_dir, cid, state="error", stage="error", error=str(exc))
    return rec
