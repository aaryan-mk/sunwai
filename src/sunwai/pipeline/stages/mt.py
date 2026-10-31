"""Translate segments into the target language with source-span routing."""

from __future__ import annotations

from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.stage import StagePolicy
from sunwai.pipeline.stages.base import EngineStage, Segments, engine


class MtStage(EngineStage[Segments, Segments]):
    """Translate segments into the target language with source-span routing."""

    name = "mt"
    policy = StagePolicy(timing_key="mt")

    def run(self, value: Segments, ctx: ChunkContext) -> Segments:
        """Translate segments into the target language with source-span routing."""
        return engine().stage_mt(ctx.legacy_args(), ctx.work_dir, value)  # type: ignore[no-any-return]
