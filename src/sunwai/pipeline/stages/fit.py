"""Rewrite translations so their synthesised speech fits the source slot."""

from __future__ import annotations

from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.stage import StagePolicy
from sunwai.pipeline.stages.base import EngineStage, Segments, engine


class FitStage(EngineStage[Segments, Segments]):
    """Duration-aware rewrite (LLM) with script, word-split, length and meaning guards."""

    name = "fit"
    policy = StagePolicy(timing_key="fit")

    def run(self, value: Segments, ctx: ChunkContext) -> Segments:
        """Fit each segment to its duration budget, budgeting per assigned voice."""
        from sunwai.pipeline.fitting.fit_stage import stage_fit

        args = ctx.legacy_args()
        args.pick_voice = engine().pick_voice  # the fit stage budgets per voice
        return stage_fit(args, ctx.work_dir, value)  # type: ignore[no-any-return]
