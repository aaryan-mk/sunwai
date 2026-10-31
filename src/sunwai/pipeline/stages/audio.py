"""Extract the 16 kHz and 44.1 kHz audio representations."""

from __future__ import annotations

from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.stage import StagePolicy
from sunwai.pipeline.stages.base import EngineStage, engine


class AudioStage(EngineStage[None, None]):
    """Extract the 16 kHz and 44.1 kHz audio representations."""

    name = "audio"
    policy = StagePolicy(timing_key="audio")

    def run(self, value: None, ctx: ChunkContext) -> None:
        """Extract the 16 kHz and 44.1 kHz audio representations."""
        engine().stage_audio(ctx.work_dir)
        return None
