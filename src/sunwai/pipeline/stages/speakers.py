"""Attach speaker identity and voice assignments to slots."""

from __future__ import annotations

from typing import Any

from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.stage import StagePolicy
from sunwai.pipeline.stages.base import EngineStage, engine


class SpeakersStage(EngineStage[list[dict[str, Any]], dict[str, Any]]):
    """Attach speaker identity and voice assignments to slots."""

    name = "speakers"
    policy = StagePolicy(timing_key="speakers")

    def run(self, value: list[dict[str, Any]], ctx: ChunkContext) -> dict[str, Any]:
        """Attach speaker identity and voice assignments to slots."""
        return engine().stage_speakers(ctx.work_dir, value)  # type: ignore[no-any-return]
