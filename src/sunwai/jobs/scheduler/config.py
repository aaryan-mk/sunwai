"""Environment-driven switches of the add-language scheduler."""

from __future__ import annotations

import os

_OFF_VALUES = ("0", "false", "no", "off", "n", "f", "")


def env_int(name: str, default: int, lo: int = 0, hi: int = 10**9) -> int:
    """Read an integer environment variable, falling back to ``default`` when unset, malformed or out of range.

    Args:
        name: Environment variable name.
        default: Value returned when the variable is missing, unparsable or outside ``[lo, hi]``.
        lo: Smallest accepted value.
        hi: Largest accepted value.

    Returns:
        The parsed value or ``default``.
    """
    try:
        value = int(float(os.environ.get(name, "").strip()))
    except (ValueError, OverflowError):
        return default
    return value if lo <= value <= hi else default


def enabled() -> bool:
    """Return the kill switch state: on by default, ``DUB_ADD_LANG=0/false/off/no`` turns it off."""
    return os.environ.get("DUB_ADD_LANG", "1").strip().lower() not in _OFF_VALUES


def lead_ms() -> int:
    """Return how far ahead of the playhead (ms) a new track is prepared first."""
    return env_int("DUB_LANG_LEAD_MS", 20000, 0, 600000)


def max_inflight() -> int:
    """Return the global cap on added-language tracks that may be preparing at once."""
    return env_int("DUB_MAX_ADDLANG_INFLIGHT", 1, 1, 16)


def orphan_grace_s() -> int:
    """Return the age (s) after which a scheduled track of a dead job stops counting toward the cap."""
    return env_int("DUB_ADDLANG_ORPHAN_S", 1800, 60, 86400)
