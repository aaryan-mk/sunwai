"""Monotonic deadlines and cooperative cancellation."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from sunwai.core.errors import Cancelled, DeadlineExceeded


@dataclass(frozen=True)
class Deadline:
    """An absolute point on the monotonic clock.

    Attributes:
        expires_at: ``time.monotonic()`` value at which the budget is spent; ``None`` means unbounded.
    """

    expires_at: float | None = None

    @classmethod
    def after(cls, seconds: float) -> Deadline:
        """Return a deadline ``seconds`` from now."""
        return cls(time.monotonic() + seconds)

    def remaining_s(self) -> float:
        """Return seconds left (``inf`` when unbounded, never negative)."""
        if self.expires_at is None:
            return float("inf")
        return max(0.0, self.expires_at - time.monotonic())

    def check(self) -> None:
        """Raise :class:`DeadlineExceeded` when the budget is spent."""
        if self.remaining_s() <= 0.0:
            raise DeadlineExceeded("deadline exceeded")


@dataclass
class CancellationToken:
    """Cooperative cancellation flag shared between a caller and a running stage."""

    _event: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        """Request cancellation."""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """True once :meth:`cancel` was called."""
        return self._event.is_set()

    def check(self) -> None:
        """Raise :class:`Cancelled` if cancellation was requested."""
        if self.cancelled:
            raise Cancelled("cancelled")
