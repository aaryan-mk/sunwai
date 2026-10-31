"""Commit paths: apply a candidate or undo a revision, each atomically with full rollback on failure."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sunwai.control.expression import engine_bridge
from sunwai.control.expression.errors import ExprError, enabled
from sunwai.control.expression.fsutil import read_json, unlink_quietly, write_bytes_atomic, write_json_atomic
from sunwai.control.expression.mixsnap import MixSnapshot, discard_mix_snapshot, restore_mix, snapshot_mix
from sunwai.control.expression.revisions import (
    Revision,
    Segment,
    active_wav,
    append_revision,
    chunk_audio_hash,
    expr_dir,
    find_segment,
    head_revision,
    load_segments,
    make_record,
    read_revisions,
    revision_wav,
    segment_lock,
    snapshot_audio,
    state_fingerprint,
    update_operation,
)

Remix = Callable[[], Any]


class LiveEditTransaction:
    """Captures everything a live edit overwrites (segment audio, ``segments.json`` bytes, mix files) so it can be restored."""

    def __init__(self, out_dir: Path, seg_id: str | int) -> None:
        """Snapshot the live state of segment ``seg_id`` in ``out_dir``."""
        self.out_dir, self.seg_id = out_dir, seg_id
        self.live = active_wav(out_dir, seg_id)
        self.segments_bytes = (out_dir / "segments.json").read_bytes()
        self._backup = expr_dir(out_dir) / f"seg{seg_id}.rollback.wav"
        if self.live.exists():
            shutil.copyfile(self.live, self._backup)
        self._mix: MixSnapshot = snapshot_mix(out_dir)

    def install_audio(self, source: str | Path, tag: str) -> None:
        """Atomically replace the live audio with ``source``."""
        tmp = self.live.with_name(f".{self.live.stem}.{tag}{os.getpid()}.wav")
        shutil.copyfile(source, tmp)
        os.replace(tmp, self.live)

    def rollback(self, orphan_revision: int) -> None:
        """Put the audio, ``segments.json`` and mix files back and delete the audio snapshot of ``orphan_revision``."""
        if self._backup.exists():
            os.replace(self._backup, self.live)
        else:
            unlink_quietly(self.live)  # no active wav existed before: don't leave the newly installed one
        write_bytes_atomic(self.out_dir / "segments.json", self.segments_bytes)
        restore_mix(self.out_dir, self._mix)
        unlink_quietly(revision_wav(self.out_dir, self.seg_id, orphan_revision))

    def commit(self) -> None:
        """Drop the backups after the change has been durably recorded."""
        discard_mix_snapshot(self._mix)
        unlink_quietly(self._backup)


def _check_head(head: Revision, expected_revision: int) -> None:
    if head["revision"] != int(expected_revision):
        raise ExprError(409, f"stale: head {head['revision']}, expected {expected_revision}")


def _apply_to_segment(seg: Segment, cand: dict[str, Any]) -> None:
    """Write the candidate's text and style into the segment; subtitles stay faithful unless a validated wording patch exists."""
    display = (cand.get("plan") or {}).get("wording_patch") or seg["tgt_text"]
    seg.update(
        tgt_text=display,
        tts_text=cand["spoken_text"],
        tts_src=display,
        tts_by="m2",
        style=cand["style"],
        style_src=("m2" if cand["style"] else None),
        tempo=cand["tempo_factor"],
        edited=True,
        overflow=False,
        overflow_s=0.0,
        expression={"intent": cand["intent"], "emotion": cand["style"], "confidence": None, "provenance": "button"},
    )


def apply_candidate(
    out_dir: str | Path,
    seg_id: str | int,
    cand_id: str,
    expected_revision: int,
    *,
    lang: str | None = None,
    remix: Remix | None = None,
) -> Revision:
    """Commit a ready candidate: swap the live audio, remix the chunk and append a new ``expression`` revision.

    Everything is rolled back if any step fails; the revision append is the commit point.

    Args:
        out_dir: Chunk directory.
        seg_id: Segment id.
        cand_id: Candidate id from a preview.
        expected_revision: Head revision the caller saw (compare-and-swap).
        lang: Language code (defaults to the candidate's).
        remix: Remix callback override (tests).

    Returns:
        The new revision record.

    Raises:
        ExprError: 503 when disabled, 404 for an unknown candidate, 409 when stale, not ready or already applied.
    """
    if not enabled():
        raise ExprError(503, "expression disabled")
    out_dir = Path(out_dir)
    cand_path = expr_dir(out_dir) / "cands" / f"{cand_id}.json"
    cand = read_json(cand_path)
    if not cand:
        raise ExprError(404, "candidate not found")
    lang = lang or cand["lang"]
    remix = remix or (lambda: engine_bridge.default_remix(out_dir, lang))
    head_revision(out_dir, lang, seg_id)
    with segment_lock(out_dir, seg_id):
        head = read_revisions(out_dir, seg_id)[-1]
        segments = load_segments(out_dir)
        seg = find_segment(segments, seg_id)
        stale = cand["base_revision"] != head["revision"] or head["state_fp"] != state_fingerprint(out_dir, seg)
        if head["revision"] != int(expected_revision) or stale:
            raise ExprError(
                409,
                f"stale: head {head['revision']}, expected {expected_revision}, candidate base {cand['base_revision']}",
            )
        if cand["status"] != "ready":
            raise ExprError(409, f"candidate not ready ({cand['status']})")
        if cand.get("applied"):
            raise ExprError(409, "candidate already applied")
        txn = LiveEditTransaction(out_dir, seg_id)
        number = head["revision"] + 1
        try:
            txn.install_audio(cand["audio_path"], "expr")
            _apply_to_segment(seg, cand)
            write_json_atomic(out_dir / "segments.json", segments)
            remix()
            # everything below must commit before the snapshot is dropped; append_revision is the commit point
            rec = make_record(
                out_dir,
                seg,
                lang,
                number,
                head["revision"],
                "expression",
                {
                    "chunk_audio_hash": chunk_audio_hash(out_dir),
                    "undo_target": head["revision"],
                    "candidate_id": cand_id,
                    "job_id": cand.get("job_id"),
                },
            )
            snapshot_audio(out_dir, seg_id, number)
            write_json_atomic(cand_path, dict(cand, applied=number))
            update_operation(
                out_dir,
                cand_id,
                state="running",
                stage="lip-syncing",
                revision=number,
                audio_ready=True,
                lipsync_ready=False,
            )
            append_revision(out_dir, seg_id, rec)
        except BaseException:
            txn.rollback(number)
            _restore_candidate_and_operation(out_dir, cand_path, cand, cand_id)
            raise
        txn.commit()
        return rec


def _restore_candidate_and_operation(out_dir: Path, cand_path: Path, cand: dict[str, Any], cand_id: str) -> None:
    """Best-effort bookkeeping after a rolled-back apply: candidate becomes appliable again, operation shows the rollback."""
    try:
        write_json_atomic(cand_path, cand)
    except Exception:
        pass
    try:
        update_operation(
            out_dir, cand_id, state="error", stage="rolled back", lipsync_ready=False, error="apply rolled back"
        )
    except Exception:
        pass


def _restore_segment_from(seg: Segment, target: Revision) -> None:
    """Restore text, style and expression of a segment from an earlier revision record."""
    for key in ("tts_text", "tts_src", "tts_by"):
        seg.pop(key, None)
    seg.update({k: target[k] for k in ("tts_text", "tts_src", "tts_by") if target.get(k)})
    seg.update(
        tgt_text=target["display_text"],
        style=target["style"],
        style_src=target.get("style_src"),
        tempo=target["tempo_factor"],
        expression=target["expression"],
    )


def undo_revision(
    out_dir: str | Path,
    seg_id: str | int,
    expected_revision: int,
    *,
    lang: str | None = None,
    remix: Remix | None = None,
) -> Revision:
    """Create a NEW revision that restores the audio, text and style of the head's ``undo_target``. Undo of an undo is a redo.

    Args:
        out_dir: Chunk directory.
        seg_id: Segment id.
        expected_revision: Head revision the caller saw (compare-and-swap).
        lang: Language code (defaults to ``"xx"``).
        remix: Remix callback override (tests).

    Returns:
        The new revision record.

    Raises:
        ExprError: 409 when stale, when there is nothing to undo, or when the prior assets are gone.
    """
    out_dir = Path(out_dir)
    lang = lang or "xx"
    remix = remix or (lambda: engine_bridge.default_remix(out_dir, lang))
    head_revision(out_dir, lang, seg_id)
    with segment_lock(out_dir, seg_id):
        revs = read_revisions(out_dir, seg_id)
        head = revs[-1]
        _check_head(head, expected_revision)
        target_number = head.get("undo_target")
        if target_number is None:
            raise ExprError(409, "nothing to undo")
        target = next(r for r in revs if r["revision"] == target_number)
        source = revision_wav(out_dir, seg_id, target_number)
        if not source.exists():
            raise ExprError(409, "prior assets missing")
        segments = load_segments(out_dir)
        seg = find_segment(segments, seg_id)
        txn = LiveEditTransaction(out_dir, seg_id)
        number = head["revision"] + 1
        try:
            txn.install_audio(source, "undo")
            _restore_segment_from(seg, target)
            write_json_atomic(out_dir / "segments.json", segments)
            remix()
            rec = make_record(
                out_dir,
                seg,
                lang,
                number,
                head["revision"],
                "undo",
                {
                    "chunk_audio_hash": chunk_audio_hash(out_dir),
                    "undo_target": head["revision"],
                    "restores": target_number,
                },
            )
            snapshot_audio(out_dir, seg_id, number)
            append_revision(out_dir, seg_id, rec)
        except BaseException:
            txn.rollback(number)
            raise
        txn.commit()
        return rec
