"""Execution of per-line actions: edit / undo / redo through the host, apply a candidate, or preview a delivery change."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from sunwai.control import hooks as ctl_hooks
from sunwai.control.codec import Record, utcnow
from sunwai.control.deps import ControlDeps
from sunwai.control.errors import CtlError
from sunwai.control.hooks import NotImplementedHook
from sunwai.control.service.claims import SynthClaims
from sunwai.control.service.confirmer import Confirmer
from sunwai.control.service.head import spoken_text
from sunwai.control.service.opstate import OperationRecorder
from sunwai.control.service.request import ActionRequest
from sunwai.control.service.watchers import watch_lipsync
from sunwai.control.service.worker import CancelState, lock_chunk, run_expression_mutation
from sunwai.control.store import JobStore

EDIT_TYPES = ("edit_line", "undo_line", "redo_line")
EXPRESSION_KEYS = ("m2_applied", "m2_candidate_id", "m2_revision", "chunk_audio_hash")
MAX_ERROR_CHARS = 300

Spawn = Callable[[Coroutine[Any, Any, None]], None]


@dataclass
class SegmentJob:
    """Everything one per-line operation needs.

    Attributes:
        job_dir: Job directory.
        job_id: Job id.
        store: The job's store.
        op: The operation record being driven.
        request: The validated action request.
        head: The head revision the action was based on.
        key: Store key of the segment.
        lang: Language of the segment.
        chunk: Chunk of the segment.
        token: Confirmation token held by this run, if any.
    """

    job_dir: Any
    job_id: str
    store: JobStore
    op: Record
    request: ActionRequest
    head: Record
    key: str
    lang: str
    chunk: int
    token: str | None

    @property
    def seg_id(self) -> str:
        """The segment id."""
        return str(self.request.core["segment_id"])


def history_length_or_none(seg: Record) -> int | None:
    """Return the length of a segment's edit history, or ``None`` when it has none."""
    return len(seg["history"]) if isinstance(seg.get("history"), list) else None


def candidate_revision(head: Record, out: Record, intent: str | None, tempo: float | None) -> Record:
    """Build the ``candidate`` revision of a preview from the head and the synthesis result.

    A candidate never inherits undo/redo bookkeeping from the head it was derived from.
    """
    rev = {
        k: v
        for k, v in head.items()
        if k not in ("m2_applied", "undo_of", "redo_of", "m2_revision", "chunk_audio_hash")
    }
    rev.update({k: v for k, v in out.items() if k.startswith("m2_")})
    rev.update(
        parent_revision=head["revision"],
        status="candidate",
        ts=utcnow(),
        spoken_text=out.get("spoken_text", head["spoken_text"]),
        style=out.get("style"),
        tempo_factor=out.get("tempo_factor", tempo or 1.0),
        audio_hash=out["audio_hash"],
        expression={"intent": intent, "emotion": None, "confidence": None, "provenance": "user_request"},
    )
    return rev


class SegmentRunner:
    """Runs one per-line operation to completion on the event loop, recording revisions and operation state."""

    def __init__(self, deps: ControlDeps, confirm: Confirmer, claims: SynthClaims, spawn: Spawn) -> None:
        """Bind the host, the confirmation table, the slot claims and the task spawner of the owning service."""
        self.d = deps
        self.confirm = confirm
        self.claims = claims
        self.spawn = spawn

    async def run(self, job: SegmentJob) -> None:
        """Run the operation; a cancellation is re-raised only AFTER the expression result was awaited and recorded."""
        cancel = CancelState()
        await self._run_guarded(job, cancel)
        if cancel.deferred:
            raise asyncio.CancelledError()

    async def _run_guarded(self, job: SegmentJob, cancel: CancelState) -> None:
        recorder = OperationRecorder(job.store, job.job_id)
        started = time.time()
        ok = False
        stack = (
            contextlib.AsyncExitStack()
        )  # holds the shared chunk lock across the expression call AND our revision recording
        try:
            recorder.stage(job.op, "synthesizing")
            job.store.emit(
                "op.stage",
                job_id=job.job_id,
                operation_id=job.op["operation_id"],
                action_id=job.op["action_id"],
                stage="synthesizing",
            )
            ok = await self._dispatch(job, recorder, started, stack, cancel)
        except NotImplementedHook as exc:
            recorder.fail(job.op, f"not_implemented: {exc}")
        except CtlError as exc:
            recorder.fail(job.op, f"{exc.code}: {exc.detail}")
        except asyncio.CancelledError:
            cancel.deferred = True
            if not cancel.recorded:
                recorder.fail(job.op, "cancelled")
        except BaseException as exc:  # incl. HTTPException from the legacy flow (state is rolled back by the edit flow)
            recorder.fail(job.op, f"{type(exc).__name__}: {getattr(exc, 'detail', exc)}"[:MAX_ERROR_CHARS])
        finally:
            try:
                await stack.aclose()
            finally:
                self.claims.release(job.store, job.key)
                self.confirm.done(job.token, ok)

    async def _dispatch(
        self,
        job: SegmentJob,
        recorder: OperationRecorder,
        started: float,
        stack: contextlib.AsyncExitStack,
        cancel: CancelState,
    ) -> bool:
        req = job.request
        if req.typ in EDIT_TYPES:
            return await self._edit_flow(job, recorder, started, stack, cancel)
        if req.apply_cand is not None and not req.preview_only:
            return await self._apply_flow(job, recorder, started, stack, cancel)
        return await self._preview_flow(job, recorder, started)

    # ---- edit / undo / redo
    async def _edit_flow(
        self,
        job: SegmentJob,
        recorder: OperationRecorder,
        started: float,
        stack: contextlib.AsyncExitStack,
        cancel: CancelState,
    ) -> bool:
        typ, head, store, key = job.request.typ, job.head, job.store, job.key
        source = (
            next(r for r in store.all_revs(key) if r["revision"] == head["undo_of"]) if typ == "redo_line" else None
        )
        keep: Record = {}
        if (
            typ in ("undo_line", "redo_line")
            and head.get("m2_applied")
            and (typ == "undo_line" or head.get("undo_of") is not None)
        ):
            res, keep, extra = await self._expression_undo(job, recorder, stack, cancel)
        else:
            res, extra = await self._host_edit(job, source)
        rev = self._edited_revision(job, res, keep, extra)
        try:
            rv = store.add_revision(key, rev, expect_head=job.request.expected, check=True)
        except BaseException as exc:  # the host/expression already committed: never leave it orphaned
            recorder.reconcile(job.op, f"committed but not recorded ({type(exc).__name__})")
            return False
        cancel.recorded = True
        self._finish(job, recorder, rv, started, undo=(typ == "undo_line"))
        return True

    async def _expression_undo(
        self, job: SegmentJob, recorder: OperationRecorder, stack: contextlib.AsyncExitStack, cancel: CancelState
    ) -> tuple[Record, Record, Record]:
        """Undo (or redo) through the expression history: undo of an undo is its redo."""
        head, typ = job.head, job.request.typ
        await lock_chunk(
            stack, getattr(self.d, "chunk_lock", None), job.job_id, job.lang, job.chunk
        )  # a legacy edit can never land between the commit and our record
        out = await run_expression_mutation(
            "expression_undo_start", recorder, job.op, cancel, job.job_dir, job.lang, job.chunk, job.seg_id
        )
        if not out.get("ok"):
            raise RuntimeError(out.get("error") or "undo failed")
        segs_now, _start = self.d.segments(job.job_dir, job.lang, job.chunk)
        res = {"segment": next((x for x in segs_now if str(x.get("id")) == job.seg_id), {})}
        keep = self._expression_keep(head, out)
        keep["audio_hash"] = out.get("audio_hash") or head["audio_hash"]
        extra = {"undo_of": head["revision"]} if typ == "undo_line" else {"redo_of": head.get("undo_of")}
        return res, keep, extra

    async def _host_edit(self, job: SegmentJob, source: Record | None) -> tuple[Record, Record]:
        """Edit, undo or redo through the host's legacy flow."""
        typ, head = job.request.typ, job.head
        expect = {"text": head["display_text"], "hist": head.get("live_hist")}
        if source is not None:  # redo: re-apply the edit that the last undo removed
            res = await self.d.edit(job.job_id, job.lang, job.chunk, job.seg_id, source["display_text"], expect)
            return res, {"redo_of": source["revision"]}
        if typ == "undo_line":
            return await self.d.undo(job.job_id, job.lang, job.chunk, job.seg_id, expect), {"undo_of": head["revision"]}
        return await self.d.edit(job.job_id, job.lang, job.chunk, job.seg_id, job.request.core["text"], expect), {}

    @staticmethod
    def _expression_keep(head: Record, out: Record) -> Record:
        return {
            "m2_applied": True,
            "m2_revision": out.get("m2_revision"),
            "chunk_audio_hash": out.get("chunk_audio_hash"),
            "m2_candidate_id": head.get("m2_candidate_id"),
        }

    def _edited_revision(self, job: SegmentJob, res: Record, keep: Record, extra: Record) -> Record:
        """Build the active revision that records an edit, undo or redo."""
        typ, head = job.request.typ, job.head
        seg_after = res.get("segment") or {}
        text = seg_after.get("tgt_text", job.request.core.get("text", head["display_text"]))
        rev = {k: v for k, v in head.items() if k not in EXPRESSION_KEYS}
        reuse = next(
            (
                r
                for r in reversed(job.store.all_revs(job.key))
                if r["display_text"] == text
                and r["revision"] != head["revision"]
                and r["status"] in ("active", "superseded")
            ),
            None,
        )
        audio_hash = (
            reuse["audio_hash"]
            if (typ == "undo_line" and reuse)
            else self.d.audio_hash(job.job_dir, job.lang, job.chunk, text)
        )
        rev.update(
            parent_revision=head["revision"],
            display_text=text,
            spoken_text=(spoken_text(seg_after) if keep and seg_after else text),
            status="active",
            ts=utcnow(),
            live_hist=history_length_or_none(seg_after),
            audio_hash=audio_hash,
            expression={"intent": None, "emotion": None, "confidence": None, "provenance": typ},
            subtitle_revision=head.get("subtitle_revision", 0) + 1,
        )
        rev["undo_of"] = None
        rev.update(extra)
        rev.update(keep)
        return rev

    # ---- apply a previewed candidate
    async def _apply_flow(
        self,
        job: SegmentJob,
        recorder: OperationRecorder,
        started: float,
        stack: contextlib.AsyncExitStack,
        cancel: CancelState,
    ) -> bool:
        head, store, key, exp = job.head, job.store, job.key, job.request.expected
        candidate = next(
            (r for r in store.all_revs(key) if r["revision"] == job.request.apply_cand and r["status"] == "candidate"),
            None,
        )
        if candidate is None or candidate["parent_revision"] != exp:
            raise CtlError(
                409,
                "stale_revision",
                "candidate is no longer based on the active revision",
                expected=exp,
                current=head["revision"],
            )
        await lock_chunk(stack, getattr(self.d, "chunk_lock", None), job.job_id, job.lang, job.chunk)
        out = await run_expression_mutation(
            "expression_apply_start", recorder, job.op, cancel, job.job_dir, job.lang, job.chunk, job.seg_id, candidate
        )
        if not out.get("ok"):
            raise RuntimeError(out.get("error") or "apply failed")
        segs_now, _start = self.d.segments(job.job_dir, job.lang, job.chunk)
        seg_now = next((x for x in segs_now if str(x.get("id")) == job.seg_id), {})
        rev = {k: v for k, v in candidate.items() if k not in ("undo_of", "redo_of")}
        rev.update(
            undo_of=None,
            parent_revision=head["revision"],
            status="active",
            audio_hash=out.get("audio_hash", candidate["audio_hash"]),
            display_text=seg_now.get("tgt_text", candidate["display_text"]),
            spoken_text=spoken_text(seg_now) if seg_now else candidate["spoken_text"],
            m2_applied=True,
            m2_revision=out.get("m2_revision"),
            live_hist=history_length_or_none(seg_now),
            chunk_audio_hash=out.get("chunk_audio_hash"),
            ts=utcnow(),
        )
        try:
            rv = store.add_revision(key, rev, expect_head=exp, check=True)
            store.set_status(key, candidate["revision"], "superseded")
        except BaseException as exc:
            recorder.reconcile(job.op, f"applied but not recorded ({type(exc).__name__})")
            return False
        cancel.recorded = True
        self._finish(job, recorder, rv, started)
        return True

    # ---- preview a new delivery
    async def _preview_flow(self, job: SegmentJob, recorder: OperationRecorder, started: float) -> bool:
        core = job.request.core
        intent, tempo = core.get("intent"), core.get("tempo_factor")
        out = await ctl_hooks.expression_resynth(
            job.job_dir,
            job.lang,
            job.chunk,
            job.seg_id,
            intent,
            job.head["display_text"],
            tempo,
            job.store.dir / "candidates",
        )
        if not out.get("ok"):
            raise RuntimeError(out.get("error") or "resynth failed")
        rv = job.store.add_revision(job.key, candidate_revision(job.head, out, intent, tempo))
        self._finish(job, recorder, rv, started, candidate=True)
        return True

    def _finish(
        self,
        job: SegmentJob,
        recorder: OperationRecorder,
        rv: Record,
        started: float,
        candidate: bool = False,
        undo: bool = False,
    ) -> None:
        """Announce that audio is ready; committed revisions then hand over to the lip-sync watcher."""
        recorder.audio_ready(job.op, rv, candidate)
        if candidate:
            recorder.candidate_done(job.op, rv, started)
            return
        recorder.committed(job.op, rv, undo)
        self.spawn(watch_lipsync(self.d, recorder, job.op, job.job_dir, job.lang, job.chunk, rv, started))
