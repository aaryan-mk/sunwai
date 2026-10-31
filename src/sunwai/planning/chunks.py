"""Stage 4 -- chunk plan: choose chunk boundaries in silence and write ``plan.json``.

The cut-point search is pure (VAD intervals and per-frame RMS in, boundaries out) so it can be tested without audio
files or models.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from sunwai.planning.fileio import atomic_json, json_or_none, load_json
from sunwai.planning.vad import merge_intervals
from sunwai.planning.wavio import SAMPLE_RATE, Samples, read_mono16k

log = logging.getLogger("planner")

FRAME_S = 0.02
_EPS = 1e-6
_DEFAULT_RAMP = [20.0, 25.0, 30.0, 40.0]
_MAX_CUTS = 100000

Gap = tuple[float, float]
Chunk = dict[str, Any]


def chunk_ramp() -> list[float]:
    """Sizes of the first chunks, so chunk n+1 is ready before chunk n finishes playing.

    Read from ``CHUNK_RAMP`` (comma separated seconds) or ``FIRST_CHUNK_S`` followed by 25,30,40; a malformed value
    falls back to 20,25,30,40.
    """
    try:
        first = os.environ.get("FIRST_CHUNK_S", "20")
        ramp = [float(x) for x in os.environ.get("CHUNK_RAMP", first + ",25,30,40").split(",") if x.strip()]
        assert all(0.5 < v < 1e4 for v in ramp)
        return ramp
    except Exception:
        return list(_DEFAULT_RAMP)


def gap_points(vad: list[Any], duration: float | None = None) -> list[Gap]:
    """Silence gaps between MERGED speech intervals, plus genuine leading and trailing silence.

    Returns ``[end_i, start_{i+1}]`` for consecutive intervals, ``[0, first_start]`` and ``[last_end, duration]``.
    (Pairing every consecutive sorted boundary used to return the speech intervals themselves as "gaps", so chunks
    could be cut mid-sentence.)
    """
    if not vad:
        return []
    speech = merge_intervals([(float(a), float(b)) for a, b in vad])
    gaps: list[Gap] = []
    if speech and speech[0][0] > _EPS:
        gaps.append((0.0, speech[0][0]))
    for i in range(len(speech) - 1):
        a, b = speech[i][1], speech[i + 1][0]
        if b - a > _EPS:
            gaps.append((a, b))
    if duration is not None and speech and duration - speech[-1][1] > _EPS:
        gaps.append((speech[-1][1], float(duration)))
    return gaps


def rms_windows(samples: Samples, sr: int = SAMPLE_RATE, frame: float = FRAME_S) -> list[float]:
    """RMS of every ``frame``-second window, computed block-wise so a long track is never held as float32 at once."""
    win = max(1, int(frame * sr))
    block = win * 3000  # 60 s of samples per block (a multiple of the frame length)
    rms: list[float] = []
    for start in range(0, len(samples), block):
        segment = samples[start : start + block]
        usable = (len(segment) // win) * win
        if usable:
            frames = np.asarray(segment[:usable], dtype=np.float32).reshape(-1, win)
            rms.extend(np.sqrt(np.mean(np.square(frames), axis=1)).astype(float).tolist())
        if len(segment) > usable:
            rms.append(float(np.sqrt(np.mean(np.square(segment[usable:])))))
    return rms


def lowest_rms_cut(rms: list[float], lo: float, hi: float) -> float | None:
    """Time of the centre of the quietest frame in ``[lo, hi]``, or ``None`` if the window holds no usable frame."""
    first = int(lo / FRAME_S)
    last = max(first + 1, int(hi / FRAME_S) + 1)
    best_i: int | None = None
    best_v = 0.0
    for i in range(first, min(last, len(rms))):
        value = rms[i]
        if value != value:  # NaN
            continue
        if best_i is None or value < best_v:
            best_v, best_i = value, i
    return None if best_i is None else (best_i + 0.5) * FRAME_S


def _next_cut(prev: float, size: float, duration: float, gaps: list[Gap], rms: list[float]) -> float | None:
    """The next boundary after ``prev`` for a chunk of about ``size`` seconds, or ``None`` when the walk is done."""
    lo, hi = prev + 0.75 * size, prev + 1.25 * size
    if lo >= duration - 1e-3:
        return None
    # candidates: the midpoint of every silence gap that falls inside the window; the longest gap wins
    mids = [((a + b) / 2.0, b - a) for a, b in gaps if b - a >= _EPS and lo <= (a + b) / 2.0 <= hi]
    if mids:
        mids.sort(key=lambda m: (-m[1], m[0]))
        return mids[0][0]
    cut = lowest_rms_cut(rms, lo, hi)
    return min(hi, duration) if cut is None else cut


def _boundaries(gaps: list[Gap], rms: list[float], duration: float, chunk: float) -> list[float]:
    """Walk from 0 picking cut points until the end of the audio."""
    ramp = chunk_ramp()
    boundaries: list[float] = []
    prev = 0.0
    for _ in range(_MAX_CUTS):
        if prev >= duration - _EPS:
            break
        size = min(ramp[len(boundaries)], chunk) if len(boundaries) < len(ramp) else chunk
        cut = _next_cut(prev, size, duration, gaps, rms)
        if cut is None or not (prev < cut < duration - _EPS) or cut - prev < 1e-3:
            break
        boundaries.append(round(cut, 3))
        prev = cut
    return boundaries


def _chunks_from_edges(edges: list[float], chunk: float) -> list[Chunk]:
    """Turn boundaries into ``{idx, start, dur}`` chunks, merging a remainder shorter than half a chunk backwards."""
    chunks: list[Chunk] = []
    for start, end in zip(edges, edges[1:], strict=False):
        if end - start > _EPS:
            chunks.append({"idx": len(chunks), "start": round(start, 3), "dur": round(end - start, 3)})
    if len(chunks) >= 2 and chunks[-1]["dur"] < 0.5 * chunk:
        last = chunks.pop()
        previous = chunks[-1]
        previous["dur"] = round(previous["start"] + previous["dur"] + last["dur"] - previous["start"], 3)
    for i, item in enumerate(chunks):
        item["idx"] = i
    return chunks


def make_cuts(vad: list[Any], duration: float, chunk: float, samples: Samples, sr: int = SAMPLE_RATE) -> list[Chunk]:
    """Split ``duration`` seconds of audio into chunks of about ``chunk`` seconds.

    Each cut is the midpoint of the LONGEST VAD gap whose midpoint lies in ``[prev+0.75*chunk, prev+1.25*chunk]``;
    otherwise the lowest-RMS 20 ms frame in that window. A last remainder under ``0.5*chunk`` merges back.
    """
    gaps = gap_points(vad, duration)
    rms = rms_windows(samples, sr)
    boundaries = _boundaries(gaps, rms, duration, chunk)
    return _chunks_from_edges([0.0, *boundaries, duration], chunk)


def stage_chunks(out_dir: str, chunk: float) -> dict[str, Any]:
    """Return the timing plan, computing the chunk list into ``plan.json`` when it is not there yet."""
    plan_path = os.path.join(out_dir, "plan.json")
    vad = load_json(os.path.join(out_dir, "vad.json"))
    samples = read_mono16k(os.path.join(out_dir, "source16k.wav"))
    duration = len(samples) / float(SAMPLE_RATE)
    plan = json_or_none(plan_path)
    if not isinstance(plan, dict):
        plan = {"duration": duration, "chunk_target": chunk, "chunks": []}
    if not plan.get("chunks"):  # idempotent: recompute only when absent
        cuts = make_cuts(vad, duration, chunk, samples)
        plan["chunks"] = cuts
        plan["n_chunks"] = len(cuts)
        atomic_json(plan_path, plan, indent=2)
    log.info("[4] plan: %d chunks (target %.0fs) -> plan.json", len(plan["chunks"]), plan.get("chunk_target", chunk))
    return plan
