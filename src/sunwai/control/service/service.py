"""The single validated action service. Buttons, transcript edits and chat all call :meth:`ActionService.submit`."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from sunwai.control import hooks as ctl_hooks
from sunwai.control.codec import Record, canon, utcnow
from sunwai.control.deps import ControlDeps
from sunwai.control.errors import CtlError
from sunwai.control.guard import Guard
from sunwai.control.hooks import NotImplementedHook
from sunwai.control.service.claims import SynthClaims
from sunwai.control.service.confirmer import Confirmer
from sunwai.control.service.head import HeadView, reconcile_head
from sunwai.control.service.opstate import OperationRecorder
from sunwai.control.service.request import ActionRequest, parse_request, summarise
from sunwai.control.service.segment_ops import SegmentJob, SegmentRunner
from sunwai.control.service.watchers import TrackWatcher
from sunwai.control.settings import addlang_enabled, limits
from sunwai.control.store import JobStore, store_for

Response = tuple[int, dict[str, Any]]


def new_operation(action_id: str, state: str, stage: str = "queued") -> Record:
    """Build a fresh operation record."""
    return {
        "operation_id": "op_" + uuid.uuid4().hex[:12],
        "action_id": action_id,
        "state": state,
        "stage": stage,
        "revision": None,
        "audio_ready": False,
        "lipsync_ready": False,
        "error": None,
    }


def synthesis_seconds(req: ActionRequest, head: Record | None) -> float:
    """Return the GPU seconds an action is budgeted at (0 when it needs no synthesis)."""
    if req.is_segment_action and head is not None:
        return max(0.0, (head["slot_end_ms"] - head["slot_start_ms"]) / 1000.0)
    if req.typ == "regenerate_chunk":
        return 30.0
    if req.typ == "add_language":
        return float(
            os.environ.get("DUB_CTL_ADDLANG_S", "120")
        )  # GPU work: counted against job + daily synthesis budgets
    return 0.0


def claim_key(req: ActionRequest, segment_key: str | None) -> str:
    """Return the synthesis-slot key of an action: the line, the language, or the chunk."""
    if req.is_segment_action and segment_key is not None:
        return segment_key
    return f"lang/{req.core['lang']}" if req.typ == "add_language" else f"chunk/{req.chunk}"


class ActionService:
    """Validates, authorises, budgets and executes control-plane actions for any job."""

    def __init__(self, deps: ControlDeps, guard: Guard | None = None) -> None:
        """Create the service.

        Args:
            deps: The host the actions run against.
            guard: Global admission control; ``None`` disables global caps (tests).
        """
        self.guard = guard
        self.d = deps
        self.confirm = Confirmer()
        self.claims = SynthClaims(guard)
        self.tasks: set[asyncio.Task[None]] = set()
        self._runner = SegmentRunner(deps, self.confirm, self.claims, self._spawn)

    # -------- helpers
    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def job_context(self, job_id: str) -> tuple[Path, JobStore]:
        """Return the job directory and store.

        Raises:
            CtlError: 404 ``unknown_job`` when the job does not exist.
        """
        job_dir = self.d.job_dir(job_id)
        if job_dir is None or not (job_dir / "job.json").is_file():
            raise CtlError(404, "unknown_job", "job not found")
        return job_dir, store_for(job_dir)

    def head_view(self, job_id: str, lang: str, chunk: int, seg_id: str) -> HeadView:
        """Reconcile the live segment into the revision log and return its head."""
        job_dir, store = self.job_context(job_id)
        return reconcile_head(self.d, job_dir, job_id, store, lang, chunk, seg_id)

    @staticmethod
    def release_language(job_id: str, lang: str, reason: str) -> None:
        """Leave a durable abandonment marker for a failed or cancelled add-language op; failures never propagate.

        The scheduler adapter refuses new work while the language still has queued or running chunks.
        """
        try:
            release = ctl_hooks.release_language
            if release:
                release(job_id, lang, str(reason))
        except Exception:
            pass

    def _track_watcher(self, store: JobStore, job_id: str) -> TrackWatcher:
        return TrackWatcher(
            OperationRecorder(store, job_id),
            lambda key: self.claims.release(store, key),
            self.release_language,
            self.confirm.done,
        )

    # -------- public
    async def submit(self, job_id: str, body: dict[str, Any]) -> Response:
        """Validate and start an action.

        Everything up to the idempotency reservation is synchronous (atomic on the event loop).

        Returns:
            ``(http_status, json_body)``.

        Raises:
            CtlError: For every 4xx/5xx outcome.
        """
        job_dir, store = self.job_context(job_id)
        req = parse_request(body, addlang_enabled())
        self._forget_failed_addlang_key(store, req)
        duplicate = store.idem_lookup(req.idem, req.payload_hash)
        if duplicate:
            return 200, {**duplicate["response"], "status": "duplicate"}
        view = self._segment_head(job_id, req) if req.is_segment_action else None
        caps = limits()
        token = req.token
        if req.needs_confirmation:
            if not token:
                return 200, self._propose(job_id, req, caps)
            self.confirm.hold(token, job_id, req.typ, req.bind_args, req.expected)
        else:
            token = None
        return await self._admit_and_dispatch(job_dir, store, job_id, req, view, token, caps)

    def _forget_failed_addlang_key(self, store: JobStore, req: ActionRequest) -> None:
        """Add-language is idempotent per (job, lang), but a FAILED earlier attempt may be retried under the same key."""
        if req.typ != "add_language" or req.idem not in store.idem:
            return
        old = store.actions.get(store.idem[req.idem][1])
        op_id = ((old or {}).get("response") or {}).get("operation") or {}
        if old and store.ops.get(op_id.get("operation_id"), {}).get("state") == "failed":
            store.idem.pop(req.idem, None)

    def _segment_head(self, job_id: str, req: ActionRequest) -> HeadView:
        """Reconcile and CAS-check the head a segment action is based on.

        Raises:
            CtlError: 422 without an integer expected revision, 409 when nothing can be redone or the head moved.
        """
        expected = req.expected
        if not isinstance(expected, int) or isinstance(expected, bool):
            raise CtlError(
                422, "invalid_args", "segment actions need args.lang, args.chunk and an integer expected_revision"
            )
        assert req.lang is not None and req.chunk is not None
        view = self.head_view(job_id, req.lang, req.chunk, req.core["segment_id"])
        head = view.head
        if req.typ == "redo_line":
            undone = head.get("undo_of") is not None and (head.get("expression") or {}).get("provenance") == "undo_line"
            source = (
                next((r for r in view.store.all_revs(view.key) if r["revision"] == head.get("undo_of")), None)
                if undone
                else None
            )
            if source is None:
                raise CtlError(409, "nothing_to_redo", "there is no undone edit to re-apply on this line")
        if head["revision"] != expected:
            raise CtlError(
                409,
                "stale_revision",
                "segment changed since you loaded it",
                expected=expected,
                current=head["revision"],
            )
        return view

    def _propose(self, job_id: str, req: ActionRequest, caps: dict[str, Any]) -> dict[str, Any]:
        """Issue a confirmation token and describe the proposal. Every action that commits, spends GPU or forks needs one."""
        token, expiry = self.confirm.issue(job_id, req.typ, req.bind_args, req.expected, caps["confirm_ttl_s"])
        proposal: dict[str, Any] = {
            "type": req.typ,
            "args": req.bind_args,
            "expected_revision": req.expected,
            "expires_at": expiry,
            "summary": summarise(req.typ, req.bind_args),
        }
        if req.typ == "change_voice":
            proposal["fork"] = {
                "from_job": job_id,
                "speaker_id": req.core["speaker_id"],
                "voice": req.core["voice"],
                "note": "voice plans are immutable; confirming creates a NEW job with a copied, modified plan",
            }
        kind = "fork_proposal" if req.typ == "change_voice" else "needs_confirmation"
        return {"status": kind, "confirmation_token": token, "proposal": proposal}

    async def _admit_and_dispatch(
        self,
        job_dir: Path,
        store: JobStore,
        job_id: str,
        req: ActionRequest,
        view: HeadView | None,
        token: str | None,
        caps: dict[str, Any],
    ) -> Response:
        """Budget, claim a synthesis slot, reserve the idempotency key and dispatch; undo all of it on any failure."""
        action_id = "act_" + uuid.uuid4().hex[:12]  # server-generated only: a caller-supplied action_id is ignored
        claimed: list[str] = []
        try:
            need = synthesis_seconds(req, view.head if view else None)
            if need:
                claimed.append(self._claim(store, job_id, req, view, need, caps))
            existing, in_progress = store.idem_reserve(req.idem, req.payload_hash, action_id)
            if in_progress:
                for key in claimed:
                    self.claims.release(store, key)
                self.confirm.done(token, False)
                return 200, {
                    **(existing["response"] if existing else {"action_id": in_progress}),
                    "status": "duplicate",
                }
            if need:  # charge the attempt now: failed and retried attempts cost too
                store.add_spend(synth_s=need)
                if self.guard:
                    self.guard.charge(synth_s=need)
            return await self._dispatch(job_dir, store, job_id, action_id, req, view, token)
        except BaseException:
            for key in claimed:
                self.claims.release(store, key)
            store.idem_release(req.idem, action_id)
            self.confirm.done(token, False)
            raise

    def _claim(
        self, store: JobStore, job_id: str, req: ActionRequest, view: HeadView | None, need: float, caps: dict[str, Any]
    ) -> str:
        """Check budgets and rate limits, then claim the synthesis slot; returns its key."""
        store.check_caps(caps["spend_cap_usd"], caps["synth_cap_s"], need)
        if self.guard:
            self.guard.check_daily(need_synth_s=need)
        if req.preview_only:
            store.check_preview_rate(caps["previews_per_min"])
        key = claim_key(req, view.key if view else None)
        self.claims.claim(store, key, caps["max_synth"])
        return key

    # -------- dispatch
    async def _dispatch(
        self,
        job_dir: Path,
        store: JobStore,
        job_id: str,
        action_id: str,
        req: ActionRequest,
        view: HeadView | None,
        token: str | None,
    ) -> Response:
        record: Record = {
            "action_id": action_id,
            "job_id": job_id,
            "idempotency_key": req.idem,
            "payload_hash": req.payload_hash,
            "type": req.typ,
            "args": {**req.core, "lang": req.lang, "chunk": req.chunk},
            "expected_revision": req.expected,
            "preview_only": req.preview_only,
            "ts": utcnow(),
        }
        if req.typ == "report_issue":
            return self._record_issue(store, record, req)
        if req.typ == "add_language":
            return await self._start_add_language(store, job_id, record, req, token)
        if req.typ == "regenerate_chunk":
            self.claims.release(store, f"chunk/{req.chunk}")
            raise CtlError(
                501,
                "not_implemented",
                "Regenerating a whole chunk is not available yet. You can edit or re-express individual lines instead.",
            )
        if req.typ == "change_voice":
            raise CtlError(
                501, "not_implemented", "Voice changes create a new job; confirming is not wired to job creation yet."
            )
        assert view is not None
        return self._start_segment_op(job_dir, store, job_id, record, req, view, token)

    @staticmethod
    def _record_issue(store: JobStore, record: Record, req: ActionRequest) -> Response:
        with open(store.dir / "issues.jsonl", "a", encoding="utf-8") as handle:
            handle.write(
                canon({"ts": utcnow(), "action_id": record["action_id"], "details": req.core["details"]}) + "\n"
            )
        response = {"status": "accepted", "action_id": record["action_id"], "operation": None, "revision": None}
        record["response"] = response
        store.add_action(record)
        return 200, response

    def _start_segment_op(
        self,
        job_dir: Path,
        store: JobStore,
        job_id: str,
        record: Record,
        req: ActionRequest,
        view: HeadView,
        token: str | None,
    ) -> Response:
        assert req.lang is not None and req.chunk is not None
        op = new_operation(record["action_id"], "queued")
        store.running[view.key] = op["operation_id"]
        store.put_op(op)
        seg_id = req.core["segment_id"]
        store.emit(
            "op.queued",
            job_id=job_id,
            operation_id=op["operation_id"],
            action_id=record["action_id"],
            action_type=req.typ,
            lang=req.lang,
            chunk=req.chunk,
            segment_id=seg_id,
            head_revision=view.head["revision"],
        )
        response = {
            "status": "accepted",
            "action_id": record["action_id"],
            "operation": op,
            "revision": None,
            "head_revision": view.head["revision"],
            "lang": req.lang,
            "chunk": req.chunk,
            "segment_id": seg_id,
        }
        record["response"] = response
        store.add_action(record)
        job = SegmentJob(job_dir, job_id, store, op, req, view.head, view.key, req.lang, req.chunk, token)
        self._spawn(self._runner.run(job))
        return 202, response

    async def _start_add_language(
        self, store: JobStore, job_id: str, record: Record, req: ActionRequest, token: str | None
    ) -> Response:
        action_id, lang = record["action_id"], req.core["lang"]
        generation = int(req.core.get("playback_generation", 0))
        op = new_operation(action_id, "running")
        store.put_op(op)
        lkey = f"lang/{lang}"
        try:
            result = await self.request_language(job_id, lang, req.core["from_ms"], generation)
        except BaseException:
            op.update(state="failed", stage="error", error="language request failed")
            store.put_op(op)
            self.claims.release(store, lkey)
            raise
        if result.get("obsolete"):
            self.confirm.done(token, True)
            self.claims.release(store, lkey)
            op.update(state="done", stage="complete")
            store.put_op(op)
            response = {
                "status": "obsolete",
                "action_id": action_id,
                "operation": op,
                "revision": None,
                "track": None,
                "generation": generation,
            }
            record["response"] = response
            store.add_action(record)
            return 200, response
        limit = float(os.environ.get("DUB_CTL_LANG_MAX", "900"))
        op.update(state="running", stage="preparing", kind="add_language", lang=lang, deadline=time.time() + limit)
        store.put_op(op)  # the slot stays CLAIMED until ready / failed / deadline
        response = {
            "status": "accepted",
            "action_id": action_id,
            "operation": op,
            "revision": None,
            "track": result.get("track"),
            "generation": generation,
        }
        record["response"] = response
        store.add_action(record)
        self._spawn(self._track_watcher(store, job_id).watch(op, lang, lkey, token))
        return 202, response

    async def request_language(self, job_id: str, lang: str, from_ms: int, generation: int) -> dict[str, Any]:
        """Ask the language scheduler for a track, dropping requests older than the newest playback generation seen."""
        _job_dir, store = self.job_context(job_id)
        with store.lock:
            if generation < store.lang_gen.get(lang, 0):
                return {"track": None, "obsolete": True}
            store.lang_gen[lang] = generation
        try:
            result = await ctl_hooks.request_language(job_id, lang, from_ms, generation)
        except NotImplementedHook as exc:
            raise CtlError(501, "not_implemented", str(exc)) from exc
        store.emit("language.requested", job_id=job_id, lang=lang, from_ms=from_ms, generation=generation)
        return result

    def resume_all(self, root: str | Path) -> int:
        """Called at startup (inside the loop): re-attach add-language ops that survived a restart.

        Re-claims their slots and re-attaches watchers with the remaining time of the ORIGINAL deadline. Over-cap or
        unclaimable ones are failed (terminal).

        Returns:
            The number of watchers re-attached.
        """
        root = Path(root)
        resumed = 0
        for job_dir in sorted(root.iterdir()) if root.is_dir() else []:
            if not (job_dir / "ctl" / "operations.jsonl").is_file():
                continue
            store = store_for(job_dir)
            for op in list(store.resumable):
                store.resumable.remove(op)
                resumed += self._resume_one(store, job_dir.name, op)
        return resumed

    def _resume_one(self, store: JobStore, job_id: str, op: Record) -> int:
        """Resume one add-language operation; returns 1 when a watcher was re-attached, else 0 after failing it."""
        recorder = OperationRecorder(store, job_id)
        lang, lkey = op.get("lang"), f"lang/{op.get('lang')}"
        refusal = None
        if not addlang_enabled():
            recorder.terminal(op, False, "add-language is disabled on this server")
            return 0
        if float(op.get("deadline") or 0) <= time.time():  # check BEFORE claiming anything
            refusal = ("language deadline passed while the server was down", "deadline passed while down")
        elif len(store.running) >= limits()["max_synth"]:  # per-job cap
            refusal = ("job already has a running synthesis; not resuming", "not resumed: running synthesis")
        else:
            try:
                self.claims.reclaim(store, lkey, op["operation_id"])
            except CtlError:
                refusal = ("could not re-claim the synthesis slot after restart", "could not re-claim slot")
        if refusal:
            recorder.terminal(op, False, refusal[0])
            self.release_language(job_id, lang, refusal[1])
            return 0
        self._spawn(self._track_watcher(store, job_id).watch(op, op["lang"], lkey, None, float(op["deadline"])))
        return 1
