"""Explicit, typed ordering of the dubbing stages for one chunk."""

from __future__ import annotations

from pathlib import Path

from sunwai.pipeline import runtime
from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.runner import Pipeline
from sunwai.pipeline.shared_prep import chunk_lookahead_gate
from sunwai.pipeline.stages import (
    FitStage,
    MixStage,
    MtStage,
    NumbersStage,
    SeparateStage,
    SharedPrepStage,
    TtsStage,
)
from sunwai.pipeline.stages.base import Segments
from sunwai.pipeline.voices import voice_preflight

STAGE_ORDER = ("prepare", "mt", "fit", "numbers", "tts", "separate", "mix")
"""Execution order; mirrors ``engine.main`` and is checked by a contract test."""


class DubWorkflow:
    """Dub one chunk: prepare (shared) -> MT -> fit -> numbers -> TTS -> separate -> mix.

    Each step is a :class:`~sunwai.pipeline.stage.Stage` run through :class:`~sunwai.pipeline.runner.Pipeline`, so
    cancellation, deadlines, validation and timing are applied the same way everywhere.
    """

    def __init__(self, pipeline: Pipeline | None = None) -> None:
        self._pipeline = pipeline or Pipeline()
        self._prepare = SharedPrepStage()
        self._mt = MtStage()
        self._fit = FitStage()
        self._numbers = NumbersStage()
        self._tts = TtsStage()
        self._separate = SeparateStage()
        self._mix = MixStage()

    def run(self, ctx: ChunkContext) -> Path:
        """Run the full chunk and return the path of the dubbed MP4.

        Raises:
            Cancelled, DeadlineExceeded, StageError: See :meth:`Pipeline.execute`.
        """
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        voice_preflight(ctx.request.language, ctx.work_dir)  # a bad voice plan fails the chunk up front
        runtime.args_dur = ctx.request.duration_s  # slot force-cut length
        chunk_lookahead_gate(ctx.work_dir)
        run = self._pipeline.execute
        segments: Segments = run(self._prepare, None, ctx)
        segments = run(self._mt, segments, ctx)
        segments = run(self._fit, segments, ctx)
        segments = run(self._numbers, segments, ctx)
        segments = run(self._tts, segments, ctx)
        run(self._separate, None, ctx)
        return run(self._mix, segments, ctx)
