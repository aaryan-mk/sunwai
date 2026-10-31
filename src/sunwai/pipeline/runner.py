"""Runs stages with cancellation, deadline, timing and validation applied uniformly."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Protocol, TypeVar

from sunwai.core.logging import bind
from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.stage import Stage

T = TypeVar("T")
U = TypeVar("U")
log = logging.getLogger(__name__)


class TimingSink(Protocol):
    """Receives per-stage durations."""

    def record(self, work_dir: Path, key: str, seconds: float) -> None:
        """Accumulate ``seconds`` under ``key`` for the chunk in ``work_dir``."""
        ...


class FileTimingSink:
    """Writes ``timings.json`` in the engine's existing format (via ``engine.save_timing``)."""

    def record(self, work_dir: Path, key: str, seconds: float) -> None:
        """Add ``seconds`` to ``timings.json`` under ``key``."""
        from sunwai.pipeline import engine

        engine.save_timing(work_dir, key, seconds)


class Pipeline:
    """Executes a single :class:`Stage` and applies the cross-cutting policy around it."""

    def __init__(self, timings: TimingSink | None = None) -> None:
        self._timings = timings or FileTimingSink()

    def execute(self, stage: Stage[T, U], value: T, ctx: ChunkContext) -> U:
        """Run ``stage`` on ``value``.

        Checks cancellation and the deadline before starting, validates the output, and records the duration when
        the stage's policy names a timing key.

        Raises:
            Cancelled: Cancellation was requested.
            DeadlineExceeded: The chunk's time budget is spent.
            StageError: The stage's own validation rejected its output.
        """
        ctx.cancellation.check()
        ctx.deadline.check()
        started = time.time()
        result = stage.run(value, ctx)
        stage.validate(result, ctx)
        elapsed = time.time() - started
        key = stage.policy.timing_key
        if key:
            self._timings.record(ctx.work_dir, key, elapsed)
        bind(log, stage=stage.name).info("done in %.1fs", elapsed)
        return result
