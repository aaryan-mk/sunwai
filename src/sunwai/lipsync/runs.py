"""Pure frame-interval logic: run splitting, speaker-gate runs and time-disjoint clustering.

Nothing here touches cv2, torch or the filesystem, so every function can be unit tested on plain lists.
Frame indices are half-open ``[start, end)`` intervals.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any, NamedTuple

FPS = 25
GATE_MIN_RUN_S = 0.6
GATE_BRIDGE_FRAMES = 10
GATE_RAMP_FRAMES = 5

Interval = tuple[int, int]
FaceKey = tuple[int, int] | None  # (window start frame, track id); None on the single-face path


class FaceRun(NamedTuple):
    """A run of frames in which one face track is re-animated.

    Attributes:
        key: Identity of the face track (``None`` for the single-face path).
        start: First frame (inclusive).
        end: Last frame (exclusive).
        landmarks: Frame index to 3-point landmarks of the face (``None`` entries mean "not detected").
    """

    key: FaceKey
    start: int
    end: int
    landmarks: dict[int, Any] | None = None


class Cluster(NamedTuple):
    """A time-disjoint group of runs that are rendered together.

    Attributes:
        start: First frame covered by any run (inclusive).
        end: Last frame covered by any run (exclusive).
        runs: The runs overlapping in time.
    """

    start: int
    end: int
    runs: list[FaceRun]


def split_into_runs(
    face_mask: Sequence[bool], fps: float, min_run_seconds: float = 1.0, bridge_frames: int = 3
) -> list[Interval]:
    """Turn a per-frame boolean mask into ``[start, end)`` runs.

    Contiguous True blocks are merged across gaps of at most ``bridge_frames`` and runs shorter than
    ``min_run_seconds`` are dropped.

    Args:
        face_mask: One boolean per frame.
        fps: Frame rate, used to convert the minimum duration to frames.
        min_run_seconds: Shortest run to keep.
        bridge_frames: Largest gap (in frames) that is bridged.

    Returns:
        The surviving runs in frame order.
    """
    blocks = _true_blocks(face_mask)
    merged = _bridge_blocks(blocks, bridge_frames)
    min_frames = int(min_run_seconds * fps)
    return [(start, end) for start, end in merged if end - start >= min_frames]


def _true_blocks(mask: Sequence[bool]) -> list[list[int]]:
    """Return ``[start, end)`` blocks of consecutive True values."""
    blocks: list[list[int]] = []
    start: int | None = None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            blocks.append([start, index])
            start = None
    if start is not None:
        blocks.append([start, len(mask)])
    return blocks


def _bridge_blocks(blocks: list[list[int]], max_gap: int) -> list[list[int]]:
    """Merge neighbouring blocks that are separated by at most ``max_gap`` frames."""
    merged: list[list[int]] = []
    for block in blocks:
        if merged and block[0] - merged[-1][1] <= max_gap:
            merged[-1][1] = block[1]
        else:
            merged.append(list(block))
    return merged


def split_long_run(start: int, end: int, max_frames: int) -> list[Interval]:
    """Split a run that is much longer than ``max_frames`` into near-equal pieces.

    Runs up to 1.3 x ``max_frames`` are kept whole; ``max_frames`` of 0 disables splitting.

    Args:
        start: First frame (inclusive).
        end: Last frame (exclusive).
        max_frames: Target maximum piece length.

    Returns:
        The pieces in frame order.
    """
    length = end - start
    if not max_frames or length <= max_frames * 1.3:
        return [(start, end)]
    pieces = math.ceil(length / max_frames)
    step = math.ceil(length / pieces)
    return [(first, min(end, first + step)) for first in range(start, end, step)]


def cluster_runs(runs: Sequence[FaceRun]) -> list[Cluster]:
    """Group runs into time-disjoint clusters; faces speaking at the same time share one cluster.

    Args:
        runs: Runs of any faces, in any order.

    Returns:
        Clusters ordered by time whose frame ranges never overlap.
    """
    clusters: list[Cluster] = []
    for run in sorted(runs, key=lambda r: (r[1], r[2])):
        if clusters and run[1] < clusters[-1].end:
            last = clusters[-1]
            clusters[-1] = Cluster(last.start, max(last.end, run[2]), [*last.runs, run])
        else:
            clusters.append(Cluster(run[1], run[2], [run]))
    return clusters


def gate_runs(
    runs_abs: Sequence[Interval],
    speaking_fn: Callable[[int, int], Iterable[bool]],
    *,
    fps: float = FPS,
    min_run_s: float = GATE_MIN_RUN_S,
    bridge: int = GATE_BRIDGE_FRAMES,
    ramp: int = GATE_RAMP_FRAMES,
) -> tuple[list[Interval], dict[int, float]]:
    """Restrict face runs to the frames in which the face is actually speaking.

    Frames outside the gated runs keep their original pixels; ``ramp`` frames on each side of a gated run get a
    linear cross-fade weight below 1.

    Args:
        runs_abs: Candidate runs in absolute frames.
        speaking_fn: ``(start, end)`` to a speaking mask over ``[start, end)``.
        fps: Frame rate for the minimum-duration conversion.
        min_run_s: Shortest speaking run to keep.
        bridge: Largest non-speaking gap that is bridged.
        ramp: Cross-fade length at each boundary.

    Returns:
        The gated runs and a map from absolute frame to the weight (below 1) of the generated pixels.
    """
    alpha: dict[int, float] = {}
    gated: list[Interval] = []
    for start, end in runs_abs:
        mask = [bool(flag) for flag in speaking_fn(start, end)]
        for run_start, run_end in split_into_runs(mask, fps, min_run_s, bridge):
            lead = min(ramp, run_start)
            tail = min(ramp, (end - start) - run_end)
            for j in range(lead):
                alpha[start + run_start - lead + j] = (j + 1) / (lead + 1)
            for j in range(tail):
                alpha[start + run_end + j] = 1 - (j + 1) / (tail + 1)
            gated.append((start + run_start - lead, start + run_end + tail))
    return gated, alpha


def plan_clusters(face_runs: Sequence[FaceRun], max_run_frames: int) -> list[Cluster]:
    """Split overlong runs and group all runs into time-disjoint clusters.

    Args:
        face_runs: Runs of every face of a window.
        max_run_frames: Target maximum run length (0 disables splitting).

    Returns:
        Clusters ordered by time; faces speaking at the same time share one.
    """
    pieces = [
        FaceRun(run.key, start, end, run.landmarks)
        for run in face_runs
        for start, end in split_long_run(run.start, run.end, max_run_frames)
    ]
    return cluster_runs(pieces)
