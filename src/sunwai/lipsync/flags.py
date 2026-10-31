"""Environment kill switches shared by the lip-sync modules."""

from __future__ import annotations

import os

_OFF_VALUES = ("0", "false", "off", "no", "")
_ON_VALUES = ("1", "true", "on", "yes")


def env_enabled(name: str, default: str = "1") -> bool:
    """Read a default-on flag: anything except 0/false/off/no/empty keeps the feature enabled.

    Args:
        name: Environment variable name.
        default: Value assumed when the variable is unset.

    Returns:
        False only when the variable is explicitly switched off.
    """
    return str(os.environ.get(name, default)).strip().lower() not in _OFF_VALUES


def env_opt_in(name: str) -> bool:
    """Read a default-off flag: only 1/true/on/yes enable the feature.

    Args:
        name: Environment variable name.

    Returns:
        True when the variable is explicitly switched on.
    """
    return str(os.environ.get(name, "0")).strip().lower() in _ON_VALUES
