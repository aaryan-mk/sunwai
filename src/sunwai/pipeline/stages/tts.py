"""Synthesise segments with the split and runaway-repair policies."""

from __future__ import annotations

from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.stage import StagePolicy
from sunwai.pipeline.stages.base import EngineStage, Segments, engine


class TtsStage(EngineStage[Segments, Segments]):
    """Synthesise segments with the split and runaway-repair policies."""

    name = "tts"
    policy = StagePolicy(timing_key="tts")

    def run(self, value: Segments, ctx: ChunkContext) -> Segments:
        """Synthesise segments with the split and runaway-repair policies."""
        return engine().stage_tts(ctx.legacy_args(), ctx.work_dir, value)  # type: ignore[no-any-return]
