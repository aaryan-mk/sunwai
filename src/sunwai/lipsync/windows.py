"""Speech windows: merge dub segments and pad them into the spans the lip-sync engine processes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

MERGE_GAP_S = 1.0


def create_segments(segments: Sequence[dict[str, Any]], pad: float, total_duration: float) -> list[list[float]]:
    """Merge nearby speech segments and pad them into non-overlapping windows.

    Segments less than one second apart are merged. Each merged span is padded by ``pad`` seconds on both sides,
    clamped to ``[0, total_duration]`` and trimmed so it never overlaps the previous window.

    Args:
        segments: Dicts with ``start`` and ``end`` in seconds.
        pad: Padding added on each side, in seconds.
        total_duration: Length of the chunk, in seconds.

    Returns:
        ``[start, end]`` windows in time order; empty when there are no segments.
    """
    if not segments:
        return []
    windows: list[list[float]] = []
    for start, end in _merge_close_segments(segments):
        padded_start = max(0.0, start - pad)
        padded_end = min(total_duration, end + pad)
        if windows:
            padded_start = max(padded_start, windows[-1][1])
        if padded_end > padded_start:
            windows.append([padded_start, padded_end])
    return windows


def _merge_close_segments(segments: Sequence[dict[str, Any]]) -> list[tuple[float, float]]:
    """Sort segments by start and merge those separated by less than ``MERGE_GAP_S``."""
    ordered = sorted(segments, key=lambda seg: seg["start"])
    merged: list[tuple[float, float]] = []
    current_start, current_end = ordered[0]["start"], ordered[0]["end"]
    for seg in ordered[1:]:
        if seg["start"] - current_end < MERGE_GAP_S:
            current_end = max(current_end, seg["end"])
        else:
            merged.append((current_start, current_end))
            current_start, current_end = seg["start"], seg["end"]
    merged.append((current_start, current_end))
    return merged
