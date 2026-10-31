"""Operation lifecycle: persisting operation state and announcing it on the job's event stream."""

from __future__ import annotations

import time
from typing import Any

from sunwai.control.codec import Record, canon, seg_key, utcnow
from sunwai.control.store import JobStore

TERMINAL_STAGES = {"complete": "done", "error": "failed"}


class OperationRecorder:
    """Drives operations of one job to persisted, announced states."""

    def __init__(self, store: JobStore, job_id: str) -> None:
        """Bind the job's store."""
        self.store = store
        self.job_id = job_id

    def _safe_put(self, op: Record) -> bool:
        """Persist the op, retrying once. Returns ``True`` when persisted."""
        for _ in range(2):
            try:
                self.store.put_op(op)
                return True
            except Exception:
                time.sleep(0.01)
        return False

    def stage(self, op: Record, stage: str, **fields: Any) -> None:
        """Move ``op`` to ``stage`` (``complete`` and ``error`` are terminal) and persist it."""
        op.update(stage=stage, state=TERMINAL_STAGES.get(stage, "running"), **fields)
        self.store.put_op(op)

    def terminal(self, op: Record, ok: bool, err: str | None, events: list[tuple[str, dict[str, Any]]] | None = None) -> bool:
        """Drive ``op`` to a TERMINAL persisted state even if persistence or event writes misbehave.

        Persists (retrying once), then emits events best effort. If success cannot be persisted, failure is recorded instead.

        Returns:
            Whether the operation ended in success.
        """
        if ok:
            op.update(state="done", stage="complete", error=None)
        else:
            op.update(state="failed", stage="error", error=err)
        persisted = self._safe_put(op)
        if not persisted and ok:
            op.update(state="failed", stage="error", error="completed but the completion record could not be written")
            persisted = self._safe_put(op)
            ok = False
        emitted = (events or []) if ok else [("op.error", {"operation_id": op["operation_id"], "action_id": op["action_id"], "error": op.get("error")})]
        for typ, fields in emitted:
            try:
                self.store.emit(typ, job_id=self.job_id, **fields)
            except Exception:
                pass
        return ok

    def fail(self, op: Record, err: str) -> None:
        """Mark ``op`` failed with ``err`` and announce it."""
        op.update(error=err)
        self.stage(op, "error")
        self.store.emit("op.error", job_id=self.job_id, operation_id=op["operation_id"], action_id=op["action_id"], error=err)

    def reconcile(self, op: Record, message: str) -> None:
        """Record that the host committed a change this service could not record, and fail ``op``."""
        try:
            with open(self.store.dir / "reconcile.jsonl", "a", encoding="utf-8") as handle:
                handle.write(canon({"ts": utcnow(), "operation_id": op["operation_id"], "action_id": op["action_id"], "note": message}) + "\n")
        except OSError:
            pass
        op.update(error="reconcile: " + message)
        self.stage(op, "error")
        try:
            self.store.emit("op.error", job_id=self.job_id, operation_id=op["operation_id"], action_id=op["action_id"], error=op["error"], reconcile=True)
        except Exception:
            pass

    def audio_ready(self, op: Record, rv: Record, candidate: bool) -> None:
        """Audio is ready. ``lipsync_ready`` is NEVER set here: a watcher sets it from the real lip-sync file."""
        op.update(revision=rv["revision"], audio_ready=True, lipsync_ready=False)
        self.stage(op, "audio_ready")
        key = seg_key(rv["lang"], rv["chunk_id"], rv["segment_id"])
        head = self.store.head(key)
        media = f"/api/jobs/{self.job_id}/ctl/media/%s/{rv['lang']}/{rv['chunk_id']}/{rv['segment_id']}/{rv['revision']}"
        head_no = head["revision"] if head else 0
        self.store.emit(
            "op.audio_ready", operation_id=op["operation_id"], action_id=op["action_id"], lang=rv["lang"], chunk=rv["chunk_id"],
            segment_id=rv["segment_id"], revision=rv["revision"], candidate=candidate,
            candidate_revision=rv["revision"] if candidate else None, head_revision=head["revision"] if head else None,
            audio_hash=rv["audio_hash"], audio_url=media % ("cand" if candidate else "base"),
            preview_url=(media % "cand") if candidate else None,
            base_audio_url=f"/api/jobs/{self.job_id}/ctl/media/base/{rv['lang']}/{rv['chunk_id']}/{rv['segment_id']}/{head_no}",
        )

    def committed(self, op: Record, rv: Record, undo: bool) -> None:
        """Announce a committed (or undone) revision and move to lip syncing."""
        self.store.emit("revision.undone" if undo else "revision.committed", job_id=self.job_id, revision=rv, head_revision=rv["revision"])
        self.store.emit(
            "subtitle.revision", job_id=self.job_id, lang=rv["lang"], chunk=rv["chunk_id"],
            segment_id=rv["segment_id"], subtitle_revision=rv["subtitle_revision"],
        )
        self.stage(op, "lip_syncing")

    def candidate_done(self, op: Record, rv: Record, started: float) -> None:
        """A candidate is a preview: no lip-sync is attempted, the operation completes."""
        self.stage(op, "complete")
        self.store.emit(
            "op.done", job_id=self.job_id, operation_id=op["operation_id"], action_id=op["action_id"],
            revision=rv["revision"], elapsed_s=round(time.time() - started, 2),
        )
