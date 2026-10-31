"""Per-job event log: monotonically numbered events, a bounded in-memory tail and an async wake-up for SSE consumers."""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Any

from sunwai.control.codec import Record, utcnow
from sunwai.control.jsonl import JsonlFile

EV_KEEP = int(os.environ.get("DUB_CTL_EV_KEEP", "2000"))
EV_REPLAY = int(os.environ.get("DUB_CTL_EV_REPLAY", "200"))
EV_MAX_BYTES = int(os.environ.get("DUB_CTL_EV_MAX_BYTES", str(4 << 20)))
ENVELOPE_FIELDS = ("seq", "ts", "type")


class EventLog:
    """Durable, numbered event stream of one job.

    Writers call :meth:`emit`; SSE consumers use :meth:`waiter` plus :meth:`events_after`.
    """

    def __init__(self, file: JsonlFile, job_id: str, lock: threading.RLock) -> None:
        """Load the retained tail of ``file`` and rotate it when oversized.

        Args:
            file: The ``events.jsonl`` log.
            job_id: Identifier stamped on every event.
            lock: The owning store's lock, shared so emission is atomic with state changes.
        """
        self._file = file
        self._job_id = job_id
        self._lock = lock
        self.events: list[Record] = file.read_tail(EV_KEEP)
        self.last_seq: int = self.events[-1]["seq"] if self.events else 0
        self._wakeup: tuple[asyncio.Event, asyncio.AbstractEventLoop] | None = None
        self._rotate()

    def _rotate(self) -> None:
        self._file.rotate_keeping(self.events, EV_MAX_BYTES)

    def emit(self, typ: str, **data: Any) -> Record:
        """Append an event of type ``typ``; the envelope fields (seq, ts, type) can never be overridden by ``data``."""
        with self._lock:
            self.last_seq += 1
            event: Record = {"seq": self.last_seq, "ts": utcnow(), "type": typ, "job_id": self._job_id}
            for field in ENVELOPE_FIELDS:
                data.pop(field, None)
            event.update(data)
            self._file.append(event)
            self.events.append(event)
            if len(self.events) > EV_KEEP * 2:
                del self.events[:-EV_KEEP]
                self._rotate()
            wakeup, self._wakeup = self._wakeup, None
        if wakeup is not None:
            try:
                wakeup[1].call_soon_threadsafe(wakeup[0].set)
            except Exception:
                pass
        return event

    def waiter(self) -> asyncio.Event:
        """Return the event set by the next :meth:`emit`; created lazily inside the consumer's loop."""
        with self._lock:
            if self._wakeup is None:
                self._wakeup = (asyncio.Event(), asyncio.get_running_loop())
            return self._wakeup[0]

    def events_after(self, after: int, limit: int | None = None) -> tuple[list[Record], bool]:
        """Return the events with ``seq > after`` (at most the last ``limit``) and whether the client must reset.

        A reset is needed when the client is behind what the log still retains.
        """
        limit = limit or EV_REPLAY
        with self._lock:
            after = max(0, after)
            events = [e for e in self.events if e["seq"] > after]
            reset = len(events) > limit or bool(self.events and self.events[0]["seq"] > after + 1)
            if len(events) > limit:
                events = events[-limit:]
            return events, reset
