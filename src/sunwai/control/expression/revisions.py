"""Segment revision log: the append-only history of one line's assets, kept under ``<chunk>/expr``."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from sunwai.control.expression.errors import ExprError
from sunwai.control.expression.fsutil import FileLock, read_json, sha256_file, write_json_atomic

Segment = dict[str, Any]
Revision = dict[str, Any]
NO_EXPRESSION = {"intent": None, "emotion": None, "confidence": None, "provenance": "none"}


def expr_dir(out_dir: str | Path) -> Path:
    """Return (creating it) the ``expr`` working directory of a chunk."""
    path = Path(out_dir) / "expr"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_segments(out_dir: str | Path) -> list[Segment]:
    """Read ``segments.json`` of a chunk.

    Raises:
        ExprError: 404 when the file is missing.
    """
    segments = read_json(Path(out_dir) / "segments.json")
    if segments is None:
        raise ExprError(404, "segments.json not found")
    return segments  # type: ignore[no-any-return]


def find_segment(segments: list[Segment], seg_id: str | int) -> Segment:
    """Return the segment whose id equals ``seg_id`` (compared as strings).

    Raises:
        ExprError: 404 when there is none.
    """
    found = next((s for s in segments if str(s["id"]) == str(seg_id)), None)
    if found is None:
        raise ExprError(404, f"segment {seg_id} not found")
    return found


def active_wav(out_dir: str | Path, seg_id: str | int) -> Path:
    """Return the path of the live TTS audio of a segment."""
    return Path(out_dir) / "tts" / f"{seg_id}.wav"


def revision_wav(out_dir: str | Path, seg_id: str | int, number: int) -> Path:
    """Return the path of the audio snapshot stored for revision ``number``."""
    directory = expr_dir(out_dir) / "rev"
    directory.mkdir(exist_ok=True)
    return directory / f"{seg_id}_r{number}.wav"


def spoken_text_of(seg: Segment) -> str | None:
    """Return the text actually spoken: ``tts_text`` when it belongs to the current display text, else the display text."""
    if seg.get("tts_text") and seg.get("tts_src") == seg.get("tgt_text"):
        return seg.get("tts_text")  # type: ignore[no-any-return]
    return seg.get("tgt_text")  # type: ignore[no-any-return]


def state_fingerprint(out_dir: str | Path, seg: Segment) -> str:
    """Fingerprint the text, style and audio of a segment to detect edits made outside this subsystem."""
    wav = active_wav(out_dir, seg["id"])
    parts = [seg.get("tgt_text"), seg.get("tts_text"), seg.get("style"), sha256_file(wav) if wav.exists() else None]
    return hashlib.sha1(json.dumps(parts).encode()).hexdigest()


def revisions_path(out_dir: str | Path, seg_id: str | int) -> Path:
    """Return the path of a segment's append-only revision log."""
    return expr_dir(out_dir) / f"seg{seg_id}.revs.jsonl"


def segment_lock(out_dir: str | Path, seg_id: str | int) -> FileLock:
    """Return the per-segment lock serialising head reconciliation, apply and undo."""
    return FileLock(expr_dir(out_dir) / f"seg{seg_id}.lock")


def read_revisions(out_dir: str | Path, seg_id: str | int) -> list[Revision]:
    """Read a segment's revision log.

    A crash can leave a torn (unparseable) FINAL line, which is skipped and logged; a bad line elsewhere is corruption.
    """
    path = revisions_path(out_dir, seg_id)
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    out = []
    for index, line in enumerate(lines):
        try:
            out.append(json.loads(line))
        except ValueError:
            if index == len(lines) - 1:
                logging.getLogger("expression").warning("torn final revision line skipped (%s)", path)
                break
            raise
    return out


def append_revision(out_dir: str | Path, seg_id: str | int, rec: Revision) -> None:
    """Append one record (the commit point of apply/undo), healing a torn tail first and never leaving a partial record."""
    path = revisions_path(out_dir, seg_id)
    if path.exists():
        raw = path.read_bytes()
        if raw and not raw.endswith(b"\n"):  # heal a torn tail so the new record starts on its own line
            with open(path, "ab") as handle:
                handle.truncate(raw.rfind(b"\n") + 1)
    size = path.stat().st_size if path.exists() else 0
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            with open(path, "ab") as handle:
                handle.truncate(size)
        except OSError:
            pass
        raise


def make_record(
    out_dir: str | Path,
    seg: Segment,
    lang: str,
    revision: int,
    parent: int | None,
    source: str,
    extra: dict[str, Any] | None = None,
) -> Revision:
    """Build a revision record describing ``seg`` as it is now."""
    wav = active_wav(out_dir, seg["id"])
    rec: Revision = {
        "job_id": None,
        "lang": lang,
        "chunk_id": Path(out_dir).name,
        "segment_id": seg["id"],
        "revision": revision,
        "parent_revision": parent,
        "source_revision": None,
        "speaker_id": seg.get("speaker"),
        "voice_plan_id": None,
        "slot_start_ms": int(seg["start"] * 1000),
        "slot_end_ms": int(seg["end"] * 1000),
        "raw_source_text": seg.get("src_text"),
        "display_text": seg.get("tgt_text"),
        "spoken_text": spoken_text_of(seg),
        "tts_text": seg.get("tts_text"),
        "tts_src": seg.get("tts_src"),
        "tts_by": seg.get("tts_by"),
        "expression": seg.get("expression") or dict(NO_EXPRESSION),
        "style": seg.get("style"),
        "style_src": seg.get("style_src"),
        "tempo_factor": seg.get("tempo"),
        "audio_hash": sha256_file(wav) if wav.exists() else None,
        "subtitle_revision": revision,
        "status": "active",
        "state_fp": state_fingerprint(out_dir, seg),
        "origin": source,
        "chunk_audio_hash": None,
        "ts": time.time(),
    }
    if extra:
        rec.update(extra)
    return rec


def snapshot_audio(out_dir: str | Path, seg_id: str | int, number: int) -> None:
    """Copy the live audio to the snapshot of revision ``number`` unless it already exists."""
    live = active_wav(out_dir, seg_id)
    if live.exists() and not revision_wav(out_dir, seg_id, number).exists():
        shutil.copyfile(live, revision_wav(out_dir, seg_id, number))


def chunk_audio_hash(out_dir: str | Path) -> str | None:
    """Return the hash of the chunk's mixed ``dub.wav``, or ``None`` when it does not exist."""
    dub = Path(out_dir) / "dub.wav"
    return sha256_file(dub) if dub.exists() else None


def head_revision(out_dir: str | Path, lang: str, seg_id: str | int) -> Revision:
    """Return the head revision, auto-appending an ``initial`` or ``external`` one when the segment changed outside this subsystem."""
    seg = find_segment(load_segments(out_dir), seg_id)
    with segment_lock(out_dir, seg_id):
        revs = read_revisions(out_dir, seg_id)
        if not revs:
            return _append_head(out_dir, seg, lang, seg_id, 0, None, "initial")
        if revs[-1]["state_fp"] != state_fingerprint(out_dir, seg):
            previous = revs[-1]["revision"]
            return _append_head(out_dir, seg, lang, seg_id, previous + 1, previous, "external")
        return revs[-1]


def _append_head(
    out_dir: str | Path, seg: Segment, lang: str, seg_id: str | int, number: int, parent: int | None, origin: str
) -> Revision:
    rec = make_record(out_dir, seg, lang, number, parent, origin)
    append_revision(out_dir, seg_id, rec)
    snapshot_audio(out_dir, seg_id, number)
    return rec


def update_operation(out_dir: str | Path, op_id: str, **fields: Any) -> dict[str, Any]:
    """Merge ``fields`` into the operation record ``op_id`` (creating a queued one) and persist it."""
    directory = expr_dir(out_dir) / "ops"
    directory.mkdir(exist_ok=True)
    path = directory / f"{op_id}.json"
    current = read_json(
        path,
        {
            "operation_id": op_id,
            "action_id": None,
            "state": "queued",
            "stage": "queued",
            "revision": None,
            "audio_ready": False,
            "lipsync_ready": False,
            "error": None,
        },
    )
    current.update(fields)
    write_json_atomic(path, current)
    return current  # type: ignore[no-any-return]
