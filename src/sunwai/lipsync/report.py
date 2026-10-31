"""Timing and report bookkeeping for one lip-sync chunk (the ``lipsync.json`` document)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from sunwai.lipsync.runs import FaceKey

COVERAGE_WARN_BELOW = 0.8


class Timings:
    """Named wall-clock accumulators, in seconds."""

    def __init__(self) -> None:
        self._seconds: dict[str, float] = {}

    def add(self, name: str, seconds: float) -> None:
        """Add ``seconds`` to the accumulator ``name``."""
        self._seconds[name] = self._seconds.get(name, 0.0) + seconds

    def set(self, name: str, seconds: float) -> None:
        """Overwrite the accumulator ``name``."""
        self._seconds[name] = seconds

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Time the enclosed block into ``name``."""
        started = time.time()
        try:
            yield
        finally:
            self.add(name, time.time() - started)

    def get(self, name: str) -> float:
        """Seconds accumulated under ``name`` (0 when unused)."""
        return self._seconds.get(name, 0.0)

    def rounded(self) -> dict[str, float]:
        """All accumulators rounded to 10 ms."""
        return {name: round(value, 2) for name, value in self._seconds.items()}


@dataclass
class AsdReport:
    """Active-speaker-gate statistics of a chunk.

    Attributes:
        enabled: A gate was available.
        multiface: Multi-face gating was on.
        conf_th: Gate confidence threshold (None without a gate).
        frames_before: Face frames before gating.
        frames_after: Frames still lip-synced after gating.
        runs_after: Runs after gating.
        secs: Seconds spent gating.
        speech_s: Seconds of speech windows.
        face_s: Seconds with a face on screen.
        synced_s: Seconds actually lip-synced.
        tracks: Face tracks scored.
        error: Last gate error (the engine fails open).
    """

    enabled: bool
    multiface: bool
    conf_th: float | None = None
    frames_before: int = 0
    frames_after: int = 0
    runs_after: int = 0
    secs: float = 0.0
    speech_s: float = 0.0
    face_s: float = 0.0
    synced_s: float = 0.0
    tracks: int = 0
    error: str | None = None

    def finalize(self) -> dict[str, Any]:
        """Report dict with rounded seconds plus ``gated_skip_s`` and ``coverage`` (synced share of speech).

        Reporting only: coverage never blocks publication.
        """
        report: dict[str, Any] = {
            "enabled": self.enabled,
            "frames_before": self.frames_before,
            "frames_after": self.frames_after,
            "runs_after": self.runs_after,
            "secs": round(self.secs, 2),
            "speech_s": round(self.speech_s, 2),
            "face_s": round(self.face_s, 2),
            "synced_s": round(self.synced_s, 2),
            "tracks": self.tracks,
            "multiface": self.multiface,
            "gated_skip_s": round(max(0.0, self.face_s - self.synced_s), 2),
            "coverage": round(self.synced_s / self.speech_s, 3) if self.speech_s > 0 else None,
        }
        if self.conf_th is not None:
            report["conf_th"] = self.conf_th
        if self.error is not None:
            report["error"] = self.error
        return report


def coverage_warning(job_dir: str, asd: dict[str, Any]) -> str | None:
    """Warning text when a face was on screen but under 80% of the speech was lip-synced.

    Args:
        job_dir: Chunk directory, for the message.
        asd: Finalised report from :meth:`AsdReport.finalize`.

    Returns:
        The message, or None when coverage is fine.
    """
    coverage = asd["coverage"]
    if asd["face_s"] < 1.0 or coverage is None or coverage >= COVERAGE_WARN_BELOW:
        return None
    return (
        f"[engine] COVERAGE_WARN {job_dir}: synced {asd['synced_s']}s of {asd['speech_s']}s speech ({coverage:.0%}); "
        f"face on screen {asd['face_s']}s, gated-skip {asd['gated_skip_s']}s"
    )


def face_label(key: FaceKey) -> str | None:
    """Report label of a face key: ``"<window>:<track>"``, or None for the single-face path."""
    return None if key is None else f"{key[0]}:{key[1]}"


@dataclass
class ChunkReport:
    """The ``lipsync.json`` document of one chunk.

    Attributes:
        job_dir: Chunk directory.
        lang: Target language.
        cfg: Render settings used.
        frames: Frames in the chunk.
        windows: Speech windows.
        runs: Lip-synced runs.
    """

    job_dir: str
    lang: str
    cfg: dict[str, Any]
    frames: int
    windows: int
    runs: list[dict[str, Any]] = field(default_factory=list)

    def add_run(self, f0: int, f1: int, n: int, key: FaceKey, fps: int) -> None:
        """Record a lip-synced run."""
        self.runs.append(
            {"f0": f0, "f1": f1, "t0": round(f0 / fps, 2), "t1": round(f1 / fps, 2), "n": n, "face": face_label(key)}
        )

    def to_dict(
        self, asd: dict[str, Any], timings: Timings, durations: tuple[float, float], chunk_s: float, peak_gb: float
    ) -> dict[str, Any]:
        """Assemble the final document.

        Args:
            asd: Finalised gate report.
            timings: Stage timings (must contain ``total``).
            durations: ``(output, reference)`` durations in seconds.
            chunk_s: Chunk length in seconds.
            peak_gb: Peak GPU memory in GiB.

        Returns:
            The JSON-serialisable report.
        """
        runs = sorted(self.runs, key=lambda r: r["f0"])
        return {
            "job_dir": self.job_dir,
            "lang": self.lang,
            "cfg": self.cfg,
            "runs": runs,
            "frames": self.frames,
            "windows": self.windows,
            "asd": asd,
            "dur_out": round(durations[0], 3),
            "dur_ref": round(durations[1], 3),
            "timing_s": timings.rounded(),
            "synced_frames": sum(r["n"] for r in runs),
            "chunk_s": round(chunk_s, 2),
            "rtf": round(timings.get("total") / max(chunk_s, 1e-6), 3),
            "peak_gb": peak_gb,
        }
