"""Separate vocal and background stems."""

from __future__ import annotations

from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.stage import StagePolicy
from sunwai.pipeline.stages.base import EngineStage, engine


class SeparateStage(EngineStage[None, None]):
    """Separate vocal and background stems."""

    name = "separate"
    policy = StagePolicy(timing_key="separate")

    def run(self, value: None, ctx: ChunkContext) -> None:
        """Separate vocal and background stems."""
        engine().stage_separate(ctx.work_dir)
        return None
