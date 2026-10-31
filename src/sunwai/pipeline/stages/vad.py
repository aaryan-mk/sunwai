"""Detect speech intervals in the chunk."""

from __future__ import annotations

from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.stage import StagePolicy
from sunwai.pipeline.stages.base import EngineStage, engine


class VadStage(EngineStage[None, list[dict[str, float]]]):
    """Detect speech intervals in the chunk."""

    name = "vad"
    policy = StagePolicy(timing_key="vad")

    def run(self, value: None, ctx: ChunkContext) -> list[dict[str, float]]:
        """Detect speech intervals in the chunk."""
        return engine().stage_vad(ctx.work_dir)  # type: ignore[no-any-return]
