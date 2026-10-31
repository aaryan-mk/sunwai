"""Lip-sync bookkeeping and the read-only live view of a segment, used by the control plane and the workers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from sunwai.control.expression.errors import ExprError
from sunwai.control.expression.fsutil import lock_is_busy, read_json, sha256_file, write_json_atomic
from sunwai.control.expression.revisions import (
    expr_dir,
    find_segment,
    load_segments,
    read_revisions,
    state_fingerprint,
)

EXPRESSION_ORIGINS = ("expression", "undo")


def lipsync_complete(out_dir: str | Path, seg_id: str | int, revision: int, chunk_audio_hash: str) -> str:
    """Record that LatentSync finished for ``revision``.

    Stale (the revision moved on, or the chunk audio hash differs from both the head's and the ``dub.wav`` on disk) returns
    ``"stale"`` and records nothing; otherwise the binding used by :func:`lipsync_state` is written and ``"accepted"`` returned.
    """
    revs = read_revisions(out_dir, seg_id)
    if not revs:
        return "stale"
    head = revs[-1]
    dub = Path(out_dir) / "dub.wav"
    current = sha256_file(dub) if dub.exists() else None
    moved = head["revision"] != int(revision)
    head_mismatch = bool(head.get("chunk_audio_hash")) and head["chunk_audio_hash"] != chunk_audio_hash
    disk_mismatch = current is not None and current != chunk_audio_hash
    if moved or head_mismatch or disk_mismatch:
        return "stale"
    binding = {"revision": int(revision), "chunk_audio_hash": chunk_audio_hash, "ts": time.time()}
    write_json_atomic(expr_dir(out_dir) / f"seg{seg_id}.lipsync.json", binding)
    return "accepted"


def lipsync_state(out_dir: str | Path, lang: str, seg_id: str | int) -> str:
    """Report honest lip-sync readiness: ``"ready"`` or ``"pending"``.

    Ready needs ``dub_<lang>_lipsync.mp4`` newer than ``dub_<lang>.mp4`` and, when an expression revision was applied to the
    segment, a :func:`lipsync_complete` binding matching the head revision and the current ``dub.wav`` hash. Right after
    Apply it is always ``"pending"``.
    """
    chunk = Path(out_dir)
    dub, lipsynced = chunk / f"dub_{lang}.mp4", chunk / f"dub_{lang}_lipsync.mp4"
    try:
        fresh = lipsynced.stat().st_size > 0 and dub.exists() and lipsynced.stat().st_mtime > dub.stat().st_mtime
    except OSError:
        fresh = False
    revs = read_revisions(out_dir, seg_id)
    if revs and any(r.get("origin") in EXPRESSION_ORIGINS for r in revs):
        binding = read_json(expr_dir(out_dir) / f"seg{seg_id}.lipsync.json")
        current = sha256_file(chunk / "dub.wav") if (chunk / "dub.wav").exists() else None
        bound = bool(binding) and binding["revision"] == revs[-1]["revision"] and binding["chunk_audio_hash"] == current
        return "ready" if (fresh and bound) else "pending"
    return "ready" if fresh else "pending"


def live_state(out_dir: str | Path, seg_id: str | int) -> dict[str, Any]:
    """Read-only, non-blocking view for the control plane's head reconciliation.

    Returns ``{"m2_head", "owned", "in_flight"}``: the expression head revision, whether the live segment still equals the
    expression head state, and whether an apply/undo currently holds the segment lock. It never appends a revision.

    It FAILS CLOSED on every error path (busy lock is reported immediately; no locking support, an unopenable lock file, any
    read error or an unreadable segment all yield ``owned`` and ``in_flight``), so the control plane never invents a
    "legacy" revision from a state it could not inspect.
    """
    closed = {"m2_head": None, "owned": True, "in_flight": True}
    result: dict[str, Any] = {"m2_head": None, "owned": False, "in_flight": False}
    try:
        try:
            busy = lock_is_busy(expr_dir(out_dir) / f"seg{seg_id}.lock")
        except (OSError, RuntimeError):
            return closed  # cannot probe the lock: unknown => closed
        if busy:
            result["in_flight"] = True
            return result  # busy: report immediately, read nothing else
        revs = read_revisions(out_dir, seg_id)
        if revs:
            result["m2_head"] = revs[-1]["revision"]
            try:
                seg = find_segment(load_segments(out_dir), seg_id)
                result["owned"] = revs[-1]["state_fp"] == state_fingerprint(out_dir, seg)
            except ExprError:
                result["owned"], result["in_flight"] = True, True  # segment/state unreadable: unknown => closed
        return result
    except Exception:
        return {"m2_head": result.get("m2_head"), "owned": True, "in_flight": True}


def can_undo(out_dir: str | Path, seg_id: str | int) -> bool:
    """Tell whether the head revision can be undone (asset history lives in ``expr/``); route undo/redo here when True."""
    revs = read_revisions(out_dir, seg_id)
    return bool(revs) and revs[-1].get("undo_target") is not None
