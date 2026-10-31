"""Small shared value types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

JobId = NewType("JobId", str)
LanguageCode = NewType("LanguageCode", str)


@dataclass(frozen=True)
class TimeSpan:
    """A half-open interval ``[start_s, end_s)`` on the source timeline, in seconds."""

    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if self.end_s < self.start_s:
            raise ValueError(f"end_s {self.end_s} before start_s {self.start_s}")

    @property
    def duration_s(self) -> float:
        """Length of the interval in seconds."""
        return self.end_s - self.start_s

    def overlaps(self, other: TimeSpan) -> bool:
        """Return True when the two intervals share any time."""
        return self.start_s < other.end_s and other.start_s < self.end_s
