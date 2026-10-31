"""Environment-variable parsing helpers."""

from __future__ import annotations

import logging
import os


log = logging.getLogger(__name__)




def _envf(name, default, lo, hi):
    """finite, bounded float from the environment; blank/malformed/NaN/inf/out-of-range -> default."""
    try:
        v = float(os.environ.get(name, "").strip())
        if v == v and lo <= v <= hi:
            return v
    except Exception:
        pass
    return float(default)


def _env_workers(name, default):
    """Positive-int env (blank/invalid -> default); an explicit 1 is honoured."""
    try:
        v = int(os.environ.get(name, "").strip())
        return v if v >= 1 else default
    except ValueError:
        return default


def _env_num(name, default, typ):
    """Positive finite number from env; blank/invalid/non-positive -> default."""
    try:
        v = typ(os.environ.get(name, "").strip())
        return v if v > 0 and v != float("inf") else default
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Stage 7: MT
# ---------------------------------------------------------------------------
# the local indic-translate model copies the tail of a LONG Devanagari input through unchanged (Sameer c0003 seg 3,
# 27 words -> 44% Bengali). A wrong-script line used to get one identical retry and then count as a fallback. Recover first,
# under a hard per-segment time budget: (1) whole retry via the LOCAL model (skipped when routed_mt already used IT2),
# (2) <=MT_RECOVER_MAX_PIECES pieces of <=MT_PIECE_WORDS words translated with the LOCAL model, a bad piece repaired via IT2,
# (3) whole-segment IT2. EVERY accepted result must pass script_ok(strict=True); otherwise the segment is a fallback.
def _env_clamped(name: str, default, lo, hi, cast=float):
    try:
        v = cast((os.environ.get(name, "") or "").strip() or default)
    except (TypeError, ValueError):
        v = default
    return cast(max(lo, min(hi, v)))


# ---------------------------------------------------------------------------
# per-sentence TTS placement, gentle slow-down / slack pauses, refit meaning-guard hook.
# Every behaviour has an env kill switch (read at call time, parsed robustly); all of it fails open to the old path.
# Edited segments (edited:true) never use any of it.
# ---------------------------------------------------------------------------
def _flag(name: str, default: str = "1") -> bool:
    v = os.environ.get(name)
    v = default if v is None or not v.strip() else v
    return v.strip().lower() in ("1", "true", "yes", "on")
