"""Pure helpers over speaker turns (``{"start", "end", "speaker"}`` dicts)."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

Turn = dict[str, Any]


def speaker_totals(turns: Sequence[Turn]) -> dict[Any, float]:
    """Total speaking seconds per speaker id."""
    totals: dict[Any, float] = {}
    for turn in turns:
        totals[turn["speaker"]] = totals.get(turn["speaker"], 0.0) + turn["end"] - turn["start"]
    return totals


def main_speakers(totals: dict[Any, float], min_s: float) -> set[Any]:
    """Speakers with at least ``min_s`` of speech; the loudest speaker alone if nobody qualifies."""
    return {k for k, v in totals.items() if v >= min_s} or {max(totals, key=lambda k: totals[k])}


def nearest_speaker(candidates: Sequence[Turn], time_s: float) -> Any:
    """Speaker of the candidate turn temporally nearest to ``time_s`` (0 distance when it lies inside a turn)."""

    def distance(turn: Turn) -> float:
        if turn["start"] <= time_s <= turn["end"]:
            return 0.0
        return float(min(abs(turn["start"] - time_s), abs(turn["end"] - time_s)))

    return min(candidates, key=distance)["speaker"]


def first_appearance_order(turns: Sequence[Turn]) -> list[Any]:
    """Speaker ids in the order they first speak."""
    order: list[Hashable] = []
    for turn in sorted(turns, key=lambda t: t["start"]):
        if turn["speaker"] not in order:
            order.append(turn["speaker"])
    return order
