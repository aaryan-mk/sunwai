"""Append-only revision / action / operation / event store, one per job.

Files: ``<job>/ctl/{revisions,actions,operations,events}.jsonl``. All writes are append plus fsync; the latest record
for a key wins on load.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from sunwai.control.codec import Record, canon, seg_key, sha, utcnow
from sunwai.control.errors import CtlError
from sunwai.control.events import EventLog
from sunwai.control.jsonl import JsonlFile
from sunwai.control.ratelimit import SlidingWindow

__all__ = ["CtlError", "JobStore", "canon", "seg_key", "sha", "store_for", "utcnow"]

ADD_LANGUAGE = "add_language"


class JobStore:
    """Durable per-job state: segment revisions, actions, operations, events, spend and idempotency keys."""

    def __init__(self, job_dir: str | Path) -> None:
        """Open (and replay) the store under ``<job_dir>/ctl``.

        Operations that were queued or running when the previous process died are failed, except add-language
        operations still inside their deadline, which are left in :attr:`resumable` for the service to re-attach.
        """
        self.dir = Path(job_dir) / "ctl"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.revs: defaultdict[str, list[Record]] = defaultdict(list)
        self.actions: dict[str, Record] = {}
        self.idem: dict[str, tuple[str, str]] = {}  # idempotency_key -> (payload_hash, action_id)
        self.ops: dict[str, Record] = {}
        self.running: dict[str, str] = {}  # in-flight claim key -> operation_id (memory only)
        self.previews = SlidingWindow(60.0)
        self.spend: dict[str, float] = {"usd": 0.0, "synth_s": 0.0}
        self.resumable: list[Record] = []
        self.lang_gen: dict[str, int] = {}  # lang -> highest playback generation seen
        self._files = {name: JsonlFile(self.dir / f"{name}.jsonl") for name in ("revisions", "actions", "operations")}
        self._events = EventLog(JsonlFile(self.dir / "events.jsonl"), self.dir.parent.name, self.lock)
        self._load()

    # ---- replay
    def _load(self) -> None:
        self._load_revisions()
        for action in self._files["actions"].read_all():
            self.actions[action["action_id"]] = action
            if action.get("idempotency_key"):
                self.idem[action["idempotency_key"]] = (action["payload_hash"], action["action_id"])
        for op in self._files["operations"].read_all():
            self.ops[op["operation_id"]] = op
        self._recover_interrupted_ops()
        self._load_spend()

    def _load_revisions(self) -> None:
        latest: dict[tuple[str, int], Record] = {}
        for rev in self._files["revisions"].read_all():
            key = seg_key(rev["lang"], rev["chunk_id"], rev["segment_id"])
            latest[(key, rev["revision"])] = rev  # a later record for the same revision (status change) wins
        for (key, _number), rev in sorted(latest.items(), key=lambda item: item[0]):
            self.revs[key].append(rev)

    def _recover_interrupted_ops(self) -> None:
        for op in list(self.ops.values()):
            if op.get("state") not in ("queued", "running"):
                continue
            is_addlang = op.get("kind") == ADD_LANGUAGE
            if is_addlang and op.get("stage") == "preparing" and float(op.get("deadline") or 0) > time.time():
                self.resumable.append(dict(op))
                continue
            failed = dict(op)
            error = (
                "language deadline passed while the server was down" if is_addlang else "interrupted by server restart"
            )
            failed.update(state="failed", stage="error", error=error)
            self.put_op(failed)
            self.emit("op.error", operation_id=failed["operation_id"], action_id=failed.get("action_id"), error=error)

    def _load_spend(self) -> None:
        path = self.dir / "spend.json"
        if path.exists():
            try:
                self.spend.update(json.loads(path.read_text()))
            except Exception:
                pass

    # ---- revisions
    def head(self, key: str) -> Record | None:
        """Return the active head: the highest-numbered revision whose status is ``active``."""
        with self.lock:
            for rev in reversed(self.revs.get(key, [])):
                if rev.get("status") == "active":
                    return rev
            return None

    def all_revs(self, key: str) -> list[Record]:
        """Return a copy of every revision of a segment in append order."""
        with self.lock:
            return list(self.revs.get(key, []))

    def next_no(self, key: str) -> int:
        """Return the next free revision number of a segment."""
        with self.lock:
            return max([r["revision"] for r in self.revs.get(key, [])] + [-1]) + 1

    def add_revision(self, key: str, rev: Record, expect_head: int | None = None, check: bool = False) -> Record:
        """Append a revision.

        Args:
            key: Segment key.
            rev: Revision record (its number is assigned here).
            expect_head: Revision number the active head must have when ``check`` is set.
            check: Enforce the compare-and-swap on the head.

        Raises:
            CtlError: 409 ``stale_revision`` when the CAS fails.
        """
        with self.lock:
            head = self.head(key)
            current = head["revision"] if head else None
            if check and current != expect_head:
                raise CtlError(
                    409, "stale_revision", "segment changed since you loaded it", expected=expect_head, current=current
                )
            rev = dict(rev)
            rev["revision"] = self.next_no(key)
            self._files["revisions"].append(rev)
            self.revs[key].append(rev)
            return rev

    def set_status(self, key: str, revno: int, status: str) -> Record | None:
        """Change a revision's status by appending a new record with the same number; the latest record wins."""
        with self.lock:
            revs = self.revs[key]
            for i in range(len(revs) - 1, -1, -1):
                if revs[i]["revision"] == revno:
                    updated = dict(revs[i])
                    updated["status"] = status
                    updated["ts_status"] = utcnow()
                    self._files["revisions"].append(updated)
                    revs[i] = updated
                    return updated
        return None

    # ---- actions / idempotency
    def idem_lookup(self, idem_key: str, payload_hash: str) -> Record | None:
        """Return the stored action for an idempotency key, or ``None``.

        Raises:
            CtlError: 422 when the key was used with a different payload.
        """
        with self.lock:
            hit = self.idem.get(idem_key)
            if not hit:
                return None
            if hit[0] != payload_hash:
                raise CtlError(422, "idempotency_conflict", "idempotency_key reused with a different payload")
            return self.actions.get(hit[1])

    def idem_reserve(self, idem_key: str, payload_hash: str, action_id: str) -> tuple[Record | None, str | None]:
        """Atomically look up or reserve an idempotency key.

        Returns:
            ``(existing_record_or_None, in_progress_action_id_or_None)``. The caller must :meth:`idem_release` on failure.

        Raises:
            CtlError: 422 when the key was used with a different payload.
        """
        with self.lock:
            hit = self.idem.get(idem_key)
            if hit:
                if hit[0] != payload_hash:
                    raise CtlError(422, "idempotency_conflict", "idempotency_key reused with a different payload")
                return self.actions.get(hit[1]), hit[1]
            self.idem[idem_key] = (payload_hash, action_id)
            return None, None

    def idem_release(self, idem_key: str, action_id: str) -> None:
        """Drop a reservation that never produced an action."""
        with self.lock:
            hit = self.idem.get(idem_key)
            if hit and hit[1] == action_id and action_id not in self.actions:
                self.idem.pop(idem_key, None)

    def add_action(self, rec: Record) -> None:
        """Persist an action record and index its idempotency key."""
        with self.lock:
            self._files["actions"].append(rec)
            self.actions[rec["action_id"]] = rec
            if rec.get("idempotency_key"):
                self.idem[rec["idempotency_key"]] = (rec["payload_hash"], rec["action_id"])

    # ---- operations
    def put_op(self, op: Record) -> Record:
        """Persist the latest state of an operation (a timestamped copy is stored and returned)."""
        with self.lock:
            op = dict(op)
            op["ts"] = utcnow()
            self._files["operations"].append(op)
            self.ops[op["operation_id"]] = op
            return op

    # ---- events (SSE)
    @property
    def last_seq(self) -> int:
        """Sequence number of the newest event."""
        return self._events.last_seq

    @property
    def events(self) -> list[Record]:
        """The retained in-memory tail of the event log."""
        return self._events.events

    def emit(self, typ: str, **data: Any) -> Record:
        """Append an event; see :meth:`EventLog.emit`."""
        return self._events.emit(typ, **data)

    def waiter(self) -> Any:
        """Return the asyncio event set by the next emission."""
        return self._events.waiter()

    def events_after(self, after: int, limit: int | None = None) -> tuple[list[Record], bool]:
        """Return events after ``after`` and whether the client must reset; see :meth:`EventLog.events_after`."""
        return self._events.events_after(after, limit)

    # ---- limits / spend
    def check_preview_rate(self, per_min: int) -> None:
        """Count one preview against the per-minute allowance.

        Raises:
            CtlError: 429 ``rate_limited`` when the allowance is used up.
        """
        with self.lock:
            retry = self.previews.try_acquire(per_min)
        if retry is not None:
            raise CtlError(429, "rate_limited", f"at most {per_min} previews per minute", retry_after=retry)

    def add_spend(self, usd: float = 0.0, synth_s: float = 0.0) -> None:
        """Add to the job's spend and persist the totals atomically."""
        with self.lock:
            self.spend["usd"] = round(self.spend["usd"] + usd, 6)
            self.spend["synth_s"] = round(self.spend["synth_s"] + synth_s, 3)
            tmp = self.dir / "spend.json.tmp"
            tmp.write_text(json.dumps(self.spend))
            os.replace(tmp, self.dir / "spend.json")

    def check_caps(self, usd_cap: float, synth_cap_s: float, need_synth_s: float = 0.0) -> None:
        """Refuse work once the job's LLM or synthesis budget is exhausted.

        Raises:
            CtlError: 429 ``spend_cap`` when a cap is reached.
        """
        with self.lock:
            if self.spend["usd"] >= usd_cap:
                raise CtlError(429, "spend_cap", "per-job LLM spend cap reached", retry_after=3600)
            if self.spend["synth_s"] + need_synth_s > synth_cap_s:
                raise CtlError(429, "spend_cap", "per-job synthesis budget reached", retry_after=3600)


_STORES: dict[str, JobStore] = {}
_STORES_LOCK = threading.Lock()


def store_for(job_dir: str | Path) -> JobStore:
    """Return the process-wide store of a job directory, creating it on first use."""
    key = str(Path(job_dir).resolve())
    with _STORES_LOCK:
        if key not in _STORES:
            _STORES[key] = JobStore(job_dir)
        return _STORES[key]
