"""Environment-variable parsing shared by the planner stages."""

from __future__ import annotations

import logging
import math
import os

log = logging.getLogger("planner")

_FALSE_WORDS = ("0", "false", "no", "off", "n", "f")


def env_float(name: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
    """Parse a float environment variable.

    Args:
        name: Variable name.
        default: Value used when the variable is blank, non-numeric, non-finite or out of range.
        lo: Optional inclusive lower bound.
        hi: Optional inclusive upper bound.

    Returns:
        The parsed value, or ``default`` (with a warning when the variable was set but invalid).
    """
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = float(raw)
        if not math.isfinite(value) or (lo is not None and value < lo) or (hi is not None and value > hi):
            raise ValueError("out of range")
        return value
    except ValueError:
        log.warning(
            "env %s=%r is invalid (need a finite number in [%s, %s]); using default %s", name, raw, lo, hi, default
        )
        return default


def env_flag(name: str, default: bool = True) -> bool:
    """Parse a kill switch: blank gives ``default``; 0/false/no/off/n/f is off; anything else is on."""
    value = os.environ.get(name, "").strip().lower()
    return default if value == "" else value not in _FALSE_WORDS


def ffmpeg_timeout() -> float:
    """Hard time limit in seconds for one ffmpeg call (``PLANNER_FFMPEG_TIMEOUT``)."""
    return env_float("PLANNER_FFMPEG_TIMEOUT", 3600.0, lo=5.0)


def ffprobe_timeout() -> float:
    """Hard time limit in seconds for one ffprobe call (``PLANNER_FFPROBE_TIMEOUT``)."""
    return env_float("PLANNER_FFPROBE_TIMEOUT", 60.0, lo=1.0)
