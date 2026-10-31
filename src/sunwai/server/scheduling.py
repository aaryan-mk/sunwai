"""In-process concurrency primitives: keyed locks, the priority gate, job admission and viewer focus."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field

Priority = tuple[int, int, str]
PriorityFn = Callable[[str, str, int, bool], Priority]


class GateBusy(Exception):
    """Raised when too many tasks are already waiting for the gate."""


class KeyedLocks:
    """Named asyncio locks that are dropped again once nobody holds or awaits them.

    Callers always ``async with locks.get(key)`` straight away (no await between lookup and acquire), so discarding a
    free, waiter-less lock is safe and keeps memory bounded on a long-lived service.
    """

    PRUNE_AT = 2000

    def __init__(self) -> None:
        """Create an empty lock table."""
        self._locks: dict[str, asyncio.Lock] = {}

    def get(self, key: str) -> asyncio.Lock:
        """Return the lock for ``key``, creating it on first use.

        Args:
            key: Lock name, for example ``job:<id>`` or ``<id>:<lang>:<chunk>``.

        Returns:
            The shared lock for the key.
        """
        lock = self._locks.get(key)
        if lock is None:
            if len(self._locks) >= self.PRUNE_AT:
                self._prune()
            lock = self._locks[key] = asyncio.Lock()
        return lock

    def _prune(self) -> None:
        idle = [k for k, v in self._locks.items() if not v.locked() and not getattr(v, "_waiters", None)]
        for key in idle:
            self._locks.pop(key, None)


@dataclass
class _Waiter:
    job_id: str
    lang: str
    idx: int
    future: asyncio.Future[None]
    edit: bool


class PriorityGate:
    """Semaphore whose free slot goes to the highest-priority waiter instead of the oldest.

    Waiters are bounded, acquisition may time out, and a cancelled or timed-out waiter never leaks a slot.
    """

    def __init__(self, slots: int, priority: PriorityFn, max_waiters: int = 4000) -> None:
        """Create the gate.

        Args:
            slots: Number of concurrent holders.
            priority: Ranks waiters; lower sorts first.
            max_waiters: Waiters allowed before :class:`GateBusy` is raised.
        """
        self.free = slots
        self.waiters: list[_Waiter] = []
        self.max_waiters = max_waiters
        self._priority = priority

    async def acquire(self, job_id: str, lang: str, idx: int, edit: bool = False, timeout: float | None = None) -> None:
        """Take a slot, waiting in priority order when none is free.

        Args:
            job_id: Job the work belongs to.
            lang: Target language.
            idx: Chunk index.
            edit: True for a user edit, which always jumps the queue.
            timeout: Maximum seconds to wait; ``None`` or 0 waits indefinitely.

        Raises:
            GateBusy: If too many tasks are already waiting.
            asyncio.TimeoutError: If ``timeout`` elapsed.
        """
        if self.free > 0 and not self.waiters:
            self.free -= 1
            return
        if len(self.waiters) >= self.max_waiters:
            raise GateBusy("too many queued tasks")
        waiter = _Waiter(job_id, lang, idx, asyncio.get_running_loop().create_future(), edit)
        self.waiters.append(waiter)
        try:
            if timeout:
                await asyncio.wait_for(waiter.future, timeout)
            else:
                await waiter.future
        except BaseException:
            if waiter.future.done() and not waiter.future.cancelled():
                self.release()  # the slot was already handed to us: pass it on
            elif waiter in self.waiters:
                self.waiters.remove(waiter)
            raise

    def release(self) -> None:
        """Hand the slot to the best live waiter, or return it to the pool."""
        live = [w for w in self.waiters if not w.future.done()]
        self.waiters = live
        if not live:
            self.free += 1
            return
        best = min(live, key=lambda w: self._priority(w.job_id, w.lang, w.idx, w.edit))
        self.waiters.remove(best)
        best.future.set_result(None)


class JobAdmission:
    """At most ``slots`` uncached jobs run at once; the rest are queued, not rejected."""

    def __init__(self, slots: int) -> None:
        """Create the controller.

        Args:
            slots: Jobs allowed to run concurrently (at least 1).
        """
        self.slots = max(1, slots)
        self.running: list[str] = []
        self.queue: list[str] = []
        self._cv: asyncio.Condition | None = None

    def _condition(self) -> asyncio.Condition:
        if self._cv is None:
            self._cv = asyncio.Condition()
        return self._cv

    async def admit(self, job_id: str) -> None:
        """Wait in FIFO order until a slot is free, then occupy it.

        Args:
            job_id: Job asking for a slot.
        """
        cv = self._condition()
        async with cv:
            self.queue.append(job_id)
            try:
                await cv.wait_for(
                    lambda: bool(self.queue) and self.queue[0] == job_id and len(self.running) < self.slots
                )
            except BaseException:
                if job_id in self.queue:
                    self.queue.remove(job_id)
                cv.notify_all()
                raise
            self.queue.pop(0)
            self.running.append(job_id)

    async def release(self, job_id: str) -> None:
        """Free the slot (or queue place) of ``job_id`` and wake the waiters.

        Args:
            job_id: Job giving up its slot.
        """
        cv = self._condition()
        async with cv:
            if job_id in self.running:
                self.running.remove(job_id)
            if job_id in self.queue:
                self.queue.remove(job_id)
            cv.notify_all()

    def position(self, job_id: str) -> int | None:
        """Return 0 when running, the 1-based queue place when waiting, else ``None``."""
        if job_id in self.running:
            return 0
        if job_id in self.queue:
            return self.queue.index(job_id) + 1
        return None

    @property
    def busy(self) -> int:
        """Jobs currently running or waiting."""
        return len(self.running) + len(self.queue)


@dataclass
class FocusRegistry:
    """What each viewer is watching: the chunk about to play, in the language being watched, runs first.

    Attributes:
        entries: ``job_id -> (lang, chunk index, timestamp)`` as last reported by the player.
        generation: Counter bumped on every report; lets lip-sync workers notice changes.
        ttl_s: Seconds after which an entry no longer influences ordering.
    """

    entries: dict[str, tuple[str, int, float]] = field(default_factory=dict)
    generation: int = 0
    ttl_s: float = 1800.0

    def priority(self, job_id: str, lang: str, idx: int, edit: bool = False) -> Priority:
        """Rank a ``(lang, chunk)`` task of ``job_id``; lower runs first."""
        if edit:
            return (-1, 0, lang)  # a user edit always jumps the queue
        focus = self.entries.get(job_id)
        if not focus or time.time() - focus[2] > self.ttl_s:
            return (0, idx, lang)  # no (fresh) viewer: chunk-major order
        ahead = idx - focus[1]
        return (0 if lang == focus[0] else 1, ahead if ahead >= 0 else 1000 - ahead, lang)

    def record(self, job_id: str, lang: str, chunk: int) -> float:
        """Store a focus report, drop expired entries and bump the generation.

        Returns:
            The timestamp stored with the entry.
        """
        self.generation += 1
        now = time.time()
        for key in [k for k, v in self.entries.items() if now - v[2] > self.ttl_s]:
            self.entries.pop(key, None)
        self.entries[job_id] = (lang, chunk, now)
        return now
