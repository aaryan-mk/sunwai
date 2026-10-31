"""Produce normalised spoken text (numbers spelled out)."""

from __future__ import annotations

from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.stage import StagePolicy
from sunwai.pipeline.stages.base import EngineStage, Segments, engine


class NumbersStage(EngineStage[Segments, Segments]):
    """Produce normalised spoken text (numbers spelled out)."""

    name = "numbers"
    policy = StagePolicy(timing_key="numbers")

    def run(self, value: Segments, ctx: ChunkContext) -> Segments:
        """Produce normalised spoken text (numbers spelled out)."""
        return engine().stage_numbers(ctx.legacy_args(), ctx.work_dir, value)  # type: ignore[no-any-return]
