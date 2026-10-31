"""Sliding-window rate counting, pure apart from the injected clock."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable


class SlidingWindow:
    """Counts events inside a trailing window and reports when the next slot frees up.

    Not thread-safe by itself: callers hold their own lock around :meth:`try_acquire`.
    """

    def __init__(self, window_s: float = 60.0, clock: Callable[[], float] = time.time) -> None:
        """Create an empty window.

        Args:
            window_s: Length of the trailing window in seconds.
            clock: Time source, injectable for tests.
        """
        self.window_s = window_s
        self._clock = clock
        self._stamps: deque[float] = deque()

    def _expire(self, now: float) -> None:
        while self._stamps and now - self._stamps[0] > self.window_s:
            self._stamps.popleft()

    def try_acquire(self, limit: int) -> int | None:
        """Record one event unless ``limit`` events already happened inside the window.

        Args:
            limit: Maximum events allowed per window.

        Returns:
            ``None`` when the event was recorded, otherwise the seconds to wait before retrying.
        """
        now = self._clock()
        self._expire(now)
        if len(self._stamps) >= limit:
            return int(self.window_s - (now - self._stamps[0])) + 1
        self._stamps.append(now)
        return None

    def is_idle(self) -> bool:
        """Tell whether no event is left inside the window (the entry can be dropped)."""
        self._expire(self._clock())
        return not self._stamps

    def __len__(self) -> int:
        """Return the number of events currently inside the window."""
        return len(self._stamps)
