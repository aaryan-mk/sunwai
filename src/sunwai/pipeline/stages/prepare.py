"""Language-independent front half (fetch to ASR), computed once per chunk and shared across target languages."""

from __future__ import annotations

from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.stage import StagePolicy
from sunwai.pipeline.stages.base import EngineStage, Segments, engine


class SharedPrepStage(EngineStage[None, Segments]):
    """Fetch, audio, VAD, slots, speakers and ASR with cross-language sharing.

    The first language to take the per-chunk lock computes and publishes ``<job>/_prep/cNNNN``; the others copy it.
    The engine records the per-step timings itself, so this stage has no timing key.
    """

    name = "prepare"
    policy = StagePolicy(timing_key=None)

    def run(self, value: None, ctx: ChunkContext) -> Segments:
        """Return the source-language segments for the chunk."""
        return engine().prep_shared(ctx.legacy_args(), ctx.work_dir)  # type: ignore[no-any-return]
