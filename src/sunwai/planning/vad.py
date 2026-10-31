"""Stage 3 -- voice activity detection (silero) into ``vad.json``, processed in 10-minute blocks."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from sunwai.planning.fileio import atomic_json, json_or_none
from sunwai.planning.wavio import SAMPLE_RATE, Samples, read_mono16k

log = logging.getLogger("planner")

Interval = tuple[float, float]
BlockDetector = Callable[[Any], Iterable[dict[str, Any]]]


def merge_intervals(intervals: Iterable[Sequence[float]]) -> list[list[float]]:
    """Sort and merge overlapping/touching ``[start, end]`` intervals (empty or inverted ones are dropped)."""
    ordered = sorted((float(a), float(b)) for a, b in intervals if b > a)
    if not ordered:
        return []
    merged = [list(ordered[0])]
    for a, b in ordered[1:]:
        if a <= merged[-1][1] + 1e-3:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def blocked_vad(
    samples: Samples,
    total: float,
    detect: BlockDetector,
    sr: int = SAMPLE_RATE,
    block: float = 600.0,
    overlap: float = 0.5,
) -> list[list[float]]:
    """Run ``detect(block_samples)`` over ``block``-second blocks with a small overlap.

    ``detect`` returns ``[{start, end}]`` in block-LOCAL seconds. The block offset is ALWAYS added to make them
    global (block-local timestamps used to be treated as global for blocks after the first, collapsing later speech
    into the first ten minutes); overlap duplicates are removed by :func:`merge_intervals`.
    """
    intervals: list[list[float]] = []
    t0 = 0.0
    while t0 < total:
        t1 = min(total, t0 + block)
        segment = samples[int(round(t0 * sr)) : int(round(t1 * sr))]
        if segment.size < int(0.05 * sr):
            break
        for item in detect(segment):
            intervals.append([t0 + float(item["start"]), t0 + float(item["end"])])
        if t1 >= total:
            break
        t0 = max(t0 + 0.1, t1 - overlap)
    return intervals


def _silero_detector() -> BlockDetector:
    """Block detector backed by the silero VAD model (``padding`` is never passed; ``pad_ms=0``)."""
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad()

    def detect(segment: Any) -> Iterable[dict[str, Any]]:
        return get_speech_timestamps(  # type: ignore[no-any-return]
            segment,
            model,
            sampling_rate=SAMPLE_RATE,
            threshold=0.5,
            min_silence_duration_ms=100,
            min_speech_duration_ms=100,
            speech_pad_ms=0,
            return_seconds=True,
        )

    return detect


def stage_vad(out_dir: str) -> list[list[float]]:
    """Return the speech intervals of ``source16k.wav``, computing and caching ``vad.json`` if needed."""
    vad_path = os.path.join(out_dir, "vad.json")
    cached = json_or_none(vad_path)
    if isinstance(cached, list):
        log.info("[3] cached")
        return cached
    samples = read_mono16k(os.path.join(out_dir, "source16k.wav"))
    total = len(samples) / float(SAMPLE_RATE)
    log.info("[3] running silero VAD on %.3fs (%d samples)", total, len(samples))
    intervals = blocked_vad(samples, total, _silero_detector())
    vad = [[round(s, 3), round(e, 3)] for s, e in merge_intervals(intervals)]
    atomic_json(vad_path, vad, indent=2)
    log.info("[3] wrote vad.json (%d speech intervals)", len(vad))
    return vad
