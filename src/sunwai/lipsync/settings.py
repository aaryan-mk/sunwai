"""Per-chunk render settings and engine-wide switches."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

from sunwai.lipsync.flags import env_enabled


@dataclass(frozen=True)
class ChunkConfig:
    """Render settings of one chunk (defaults = ``lip_cfg_prod.json``).

    Attributes:
        steps: DDIM steps.
        gs: Classifier-free guidance scale.
        deepcache: DeepCache interval.
        batch: Full 16-frame windows per UNet batch.
        pad: Padding around speech windows, seconds.
        min_run: Shortest single-face run, seconds.
        bridge: Largest face-detection gap bridged, frames.
        audio_offset_ms: Audio offset in milliseconds (positive takes features from earlier audio).
        seed: Noise seed.
        crf: x264 quality of the final encode.
        max_run_s: Longest run rendered in one piece, seconds (0 disables splitting).
        pipeline: 1 for the pipelined renderer, 0 for the sequential one (kill switch).
    """

    steps: int = 8
    gs: float = 1.5
    deepcache: int = 3
    batch: int = 4
    pad: float = 0.3
    min_run: float = 1.0
    bridge: int = 3
    audio_offset_ms: float = 0
    seed: int = 1247
    crf: int = 18
    max_run_s: float = 20.0
    pipeline: int = 1

    @classmethod
    def from_overrides(cls, overrides: dict[str, Any]) -> ChunkConfig:
        """Build a config from the defaults plus overrides; unknown keys are reported and ignored."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(overrides) - known)
        if unknown:
            print(f"[engine] ignoring unknown lipsync options: {unknown}", flush=True)
        return cls(**{k: v for k, v in overrides.items() if k in known})

    def as_dict(self) -> dict[str, Any]:
        """The settings as a JSON-serialisable dict (stored in the chunk report)."""
        return asdict(self)


@dataclass(frozen=True)
class EngineFlags:
    """Engine-wide kill switches read once at start-up.

    Attributes:
        paste_v2: ROI paste-back with colour matching (env ``LIP_PASTE_V2``).
        multiface: Gate and sync every face of a shot (env ``LIP_MULTIFACE``).
    """

    paste_v2: bool
    multiface: bool

    @classmethod
    def from_env(cls) -> EngineFlags:
        """Read the switches from the environment."""
        return cls(paste_v2=env_enabled("LIP_PASTE_V2"), multiface=env_enabled("LIP_MULTIFACE"))
