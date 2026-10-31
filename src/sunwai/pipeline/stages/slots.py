"""Build bounded speech slots from VAD intervals."""

from __future__ import annotations

from typing import Any

from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.stage import StagePolicy
from sunwai.pipeline.stages.base import EngineStage, engine


class SlotsStage(EngineStage[list[dict[str, float]], list[dict[str, Any]]]):
    """Build bounded speech slots from VAD intervals."""

    name = "slots"
    policy = StagePolicy(timing_key="slots")

    def run(self, value: list[dict[str, float]], ctx: ChunkContext) -> list[dict[str, Any]]:
        """Build bounded speech slots from VAD intervals."""
        return engine().stage_slots(ctx.work_dir, value)  # type: ignore[no-any-return]
