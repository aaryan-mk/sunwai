"""Resource ceilings: refuse inputs that would exhaust RAM instead of letting the OOM killer decide."""

from __future__ import annotations

import os

from sunwai.planning.envvars import env_float
from sunwai.planning.errors import ResourceLimitError


def mem_available_bytes() -> int | None:
    """``min(MemAvailable, cgroup headroom)`` in bytes, or ``None`` when unknown."""
    candidates: list[int] = []
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    candidates.append(int(line.split()[1]) * 1024)
                    break
    except Exception:
        pass
    base = "/sys/fs/cgroup"
    try:
        with open(os.path.join(base, "memory.max")) as handle:
            limit = handle.read().strip()
        if limit != "max":
            with open(os.path.join(base, "memory.current")) as handle:
                current = int(handle.read().strip())
            candidates.append(max(0, int(limit) - current))
    except Exception:
        pass
    return min(candidates) if candidates else None


def check_resources(duration: float, what: str = "audio") -> None:
    """Refuse inputs that are too long or too memory-hungry to plan.

    The ceiling is ``PLANNER_MAX_DUR_S`` (default 4 h). The in-RAM working set is
    ``duration x 16 kHz x 4 B x PLANNER_MEM_FACTOR`` (default 6: pyannote waveform + torch copy + activations) and
    must stay under 80% of the available RAM.

    Raises:
        ResourceLimitError: With an actionable message instead of an OOM kill.
    """
    max_dur = env_float("PLANNER_MAX_DUR_S", 4 * 3600.0, lo=1.0)
    if duration > max_dur:
        raise ResourceLimitError(
            f"{what} is {duration / 60:.1f} min long; the planner ceiling is {max_dur / 60:.1f} min "
            f"(PLANNER_MAX_DUR_S). Dub a shorter range (start/dur)."
        )
    need = duration * 16000 * 4 * env_float("PLANNER_MEM_FACTOR", 6.0, lo=1.0)
    avail = mem_available_bytes()
    if avail is not None and need > 0.8 * avail:
        raise ResourceLimitError(
            f"{what} of {duration / 60:.0f} min needs ~{need / 2**30:.1f} GiB of RAM to plan but only "
            f"{avail / 2**30:.1f} GiB is available. Dub a shorter range (start/dur)."
        )
