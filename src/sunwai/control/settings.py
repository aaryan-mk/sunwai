"""Environment-driven limits and feature switches of the control plane (read at call time)."""

from __future__ import annotations

import os
from typing import Any

FALSE_WORDS = ("0", "false", "no", "off", "n", "f")


def env_float(name: str, default: float) -> float:
    """Read a float from the environment, falling back to ``default`` when unset."""
    return float(os.environ.get(name, default))


def limits() -> dict[str, Any]:
    """Return the per-job limits advertised to clients and enforced by the action service."""
    return {
        "max_synth": int(env_float("DUB_CTL_MAX_SYNTH", 1)),
        "previews_per_min": int(env_float("DUB_CTL_PREVIEWS_PER_MIN", 5)),
        "spend_cap_usd": env_float("DUB_CTL_SPEND_CAP_USD", 0.50),
        "synth_cap_s": env_float("DUB_CTL_SYNTH_CAP_S", 600),
        "confirm_ttl_s": int(env_float("DUB_CTL_CONFIRM_TTL", 300)),
        "max_edit_chars": int(env_float("DUB_MAX_EDIT_CHARS", 600)),
    }


def addlang_enabled() -> bool:
    """Tell whether add-language is on (default). ``DUB_CTL_ADDLANG=0/false/no/off`` turns it off; blank means on."""
    value = os.environ.get("DUB_CTL_ADDLANG")
    if value is None or not value.strip():
        return True
    return value.strip().lower() not in FALSE_WORDS
