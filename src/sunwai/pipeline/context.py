"""Immutable description of the chunk being dubbed and the environment a stage runs in."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from sunwai.core.config import Settings
from sunwai.core.deadlines import CancellationToken, Deadline


@dataclass(frozen=True)
class ChunkRequest:
    """What to dub: a source window and the target language.

    Attributes:
        url: Source video URL or local path.
        language: Target language code, e.g. ``"hi"``.
        start_s: Window start on the source timeline, seconds.
        duration_s: Window length, seconds.
    """

    url: str
    language: str
    start_s: float = 0.0
    duration_s: float = 90.0


@dataclass(frozen=True)
class ChunkContext:
    """Everything a stage needs besides its input value.

    Attributes:
        request: The chunk being processed.
        work_dir: Chunk output directory (``<job>/<lang>/cNNNN``); stages read and write artifacts here.
        settings: Process-wide settings (immutable).
        deadline: Overall time budget for the chunk.
        cancellation: Cooperative cancellation flag.
    """

    request: ChunkRequest
    work_dir: Path
    settings: Settings = field(default_factory=Settings)
    deadline: Deadline = field(default_factory=Deadline)
    cancellation: CancellationToken = field(default_factory=CancellationToken)

    def legacy_args(self) -> argparse.Namespace:
        """Return the ``argparse``-style namespace the engine's stage functions expect."""
        r = self.request
        return argparse.Namespace(
            url=r.url,
            lang=r.language,
            out=str(self.work_dir),
            start=r.start_s,
            dur=r.duration_s,
            resynth=None,
            text=None,
            verbose=False,
        )
