"""The :class:`Stage` protocol: one named, versioned unit of the dubbing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from sunwai.pipeline.context import ChunkContext

InT = TypeVar("InT", contravariant=True)
OutT = TypeVar("OutT")


@dataclass(frozen=True)
class StagePolicy:
    """Execution policy of a stage.

    Attributes:
        max_attempts: Whole-stage attempts. Keep at 1 for stages whose service clients already retry, so retries
            never multiply.
        timing_key: Key under which the runner records the stage duration in ``timings.json``; ``None`` when the
            stage records its own timings.
        caches_internally: True when the stage already skips finished work using artifacts in ``work_dir``.
    """

    max_attempts: int = 1
    timing_key: str | None = None
    caches_internally: bool = True


class Stage(Protocol, Generic[InT, OutT]):
    """A pipeline stage mapping ``InT`` to ``OutT`` for one chunk."""

    @property
    def name(self) -> str:
        """Stable stage name used in logs and timings."""
        ...

    @property
    def policy(self) -> StagePolicy:
        """Execution policy."""
        ...

    def run(self, value: InT, ctx: ChunkContext) -> OutT:
        """Execute the stage and return its output."""
        ...

    def validate(self, value: OutT, ctx: ChunkContext) -> None:
        """Raise :class:`~sunwai.core.errors.StageError` when ``value`` is not an acceptable output."""
        ...
