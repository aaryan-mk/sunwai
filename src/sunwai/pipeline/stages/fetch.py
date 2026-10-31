"""Obtain the exact source window as ``src.mp4`` and write ``meta.json``."""

from __future__ import annotations

from typing import Any

from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.stage import StagePolicy
from sunwai.pipeline.stages.base import EngineStage, engine


class FetchStage(EngineStage[None, dict[str, Any]]):
    """Obtain the exact source window as ``src.mp4`` and write ``meta.json``."""

    name = "fetch"
    policy = StagePolicy(timing_key="fetch")

    def run(self, value: None, ctx: ChunkContext) -> dict[str, Any]:
        """Obtain the exact source window as ``src.mp4`` and write ``meta.json``."""
        return engine().stage_fetch(ctx.legacy_args(), ctx.work_dir)  # type: ignore[no-any-return]
