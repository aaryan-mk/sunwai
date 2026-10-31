"""Transcribe slots and deduplicate boundary words."""

from __future__ import annotations

from typing import Any

from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.stage import StagePolicy
from sunwai.pipeline.stages.base import EngineStage, Segments, engine


class AsrStage(EngineStage[list[dict[str, Any]], Segments]):
    """Transcribe slots and deduplicate boundary words."""

    name = "asr"
    policy = StagePolicy(timing_key="asr")

    def run(self, value: list[dict[str, Any]], ctx: ChunkContext) -> Segments:
        """Transcribe slots and deduplicate boundary words."""
        return engine().stage_asr(ctx.legacy_args(), ctx.work_dir, value)  # type: ignore[no-any-return]
