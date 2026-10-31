"""Align, duck, mix and limit the final audio, then mux the chunk video."""

from __future__ import annotations

from pathlib import Path

from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.stage import StagePolicy
from sunwai.pipeline.stages.base import EngineStage, Segments, engine


class MixStage(EngineStage[Segments, Path]):
    """Align, duck, mix and limit the final audio, then mux the chunk video."""

    name = "mix"
    policy = StagePolicy(timing_key="mix")

    def run(self, value: Segments, ctx: ChunkContext) -> Path:
        """Align, duck, mix and limit the final audio, then mux the chunk video."""
        return engine().stage_mix(ctx.legacy_args(), ctx.work_dir, value)  # type: ignore[no-any-return]
