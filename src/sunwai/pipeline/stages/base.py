"""Base class for stages that delegate to the engine's existing, tested stage functions."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.stage import StagePolicy

InT = TypeVar("InT")
OutT = TypeVar("OutT")

Segments = list[dict[str, Any]]
"""Legacy segment records (one dict per spoken line); converted to typed models in a later phase."""


def engine() -> Any:
    """Import the engine lazily so importing :mod:`sunwai.pipeline` stays free of numpy/librosa/torch."""
    from sunwai.pipeline import engine as module

    return module


class EngineStage(Generic[InT, OutT]):
    """A stage whose work is done by a function of :mod:`sunwai.pipeline.engine`.

    Subclasses set ``name`` and ``policy`` and implement :meth:`run` as a one-line call into the engine. The engine
    function keeps its own artifact caching, so :attr:`StagePolicy.caches_internally` is True.
    """

    name = ""
    policy = StagePolicy()

    def validate(self, value: OutT, ctx: ChunkContext) -> None:
        """Accept the output (the engine function validates its own artifacts)."""
        return None
