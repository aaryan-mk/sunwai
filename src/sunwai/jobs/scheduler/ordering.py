"""Pure chunk arithmetic: which chunk contains a time, and in what order chunks are prepared."""

from __future__ import annotations

from typing import Any

from sunwai.jobs.scheduler.config import lead_ms

Chunk = dict[str, Any]  # {"idx": int, "start": float seconds, "dur": float seconds}


def chunk_for_ms(chunks: list[Chunk], ms: float) -> int:
    """Return the index of the chunk containing ``ms`` (clamped to the first or last chunk)."""
    if not chunks:
        return 0
    t = ms / 1000.0
    for c in chunks:
        if c["start"] <= t < c["start"] + c["dur"]:
            return int(c["idx"])
    return int(chunks[0]["idx"] if t < chunks[0]["start"] else chunks[-1]["idx"])


def runway(chunks: list[Chunk], from_ms: float, lead: int | None = None) -> list[int]:
    """Return the containing chunk plus every chunk starting within ``lead`` ms after ``from_ms``."""
    lead = lead_ms() if lead is None else lead
    if not chunks:
        return []
    k = chunk_for_ms(chunks, from_ms)
    horizon = (from_ms + lead) / 1000.0
    return [k] + [c["idx"] for c in chunks if c["idx"] > k and c["start"] < horizon]


def plan_order(chunks: list[Chunk], from_ms: float, lead: int | None = None) -> list[int]:
    """Order chunks for preparation.

    The containing chunk comes first, then the playable runway, then the rest ahead in order, then the chunks behind
    nearest-first (a viewer who seeks back sees the closest earlier chunk first).
    """
    lead = lead_ms() if lead is None else lead
    if not chunks:
        return []
    k = chunk_for_ms(chunks, from_ms)
    ahead = [c["idx"] for c in chunks if c["idx"] > k]
    run = runway(chunks, from_ms, lead)[1:]
    rest = [i for i in ahead if i not in run]
    behind = sorted((c["idx"] for c in chunks if c["idx"] < k), reverse=True)
    return [k] + run + rest + behind


def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """Merge touching or overlapping ``[start_ms, end_ms]`` intervals."""
    merged: list[list[int]] = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged
