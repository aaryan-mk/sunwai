"""Pause alignment of synthesised speech to the source timeline."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

from sunwai.pipeline.mixing.levels import MIX_KEEP_TAIL, _frame_db, _speech_bounds
from sunwai.pipeline.support.env import _envf


def _orig_sentence_resumes(vad_path: Path, s: float, e: float, min_gap: float = 0.35):
    """Chunk-time speech-resume instants after every >= min_gap VAD gap inside [s, e]."""
    try:
        iv = json.load(open(vad_path, encoding="utf-8"))
        iv = [((v["start"], v["end"]) if isinstance(v, dict) else tuple(v)) for v in iv]
    except Exception:
        return []
    iv = [(a, b) for a, b in iv if b > s - 0.01 and a < e + 0.01]
    return [a2 for (a1, b1), (a2, b2) in zip(iv, iv[1:]) if a2 - b1 > min_gap]


MIX_ALIGN_PAUSES = os.environ.get("MIX_ALIGN_PAUSES", "1") == "1"   # default ON (guarded, fail-open, capped by next-slot room)


MIX_ALIGN_MIN_DRIFT = _envf("MIX_ALIGN_MIN_DRIFT", 0.25, 0.05, 5.0)


def _align_pauses(t: np.ndarray, sr: int, lead: float, resumes_rel, slot_dur: float, room: float = None):
    """Re-lay a (trimmed) multi-sentence TTS wav so each sentence restarts where the original speaker resumed.

    t starts `lead` s before its detected onset; resumes_rel = original resume instants relative to the original onset.
    Cuts t inside its N longest internal silences (>= 0.2 s) and re-places piece k at resumes_rel[k-1] (never closer than
    0.15 s after the previous piece). Returns (new_t, info) or (t, None) when the guards fail (counts differ, relative
    positions disagree, drift already small, or the result would exceed the slot)."""
    N = len(resumes_rel)
    if N == 0:
        return t, None
    db = _frame_db(t, sr)
    on, off, act = _speech_bounds(db, 30.0)
    if on is None:
        return t, None
    # internal silence runs
    runs, i = [], on
    while i < off:
        if not act[i]:
            j = i
            while j < off and not act[j]:
                j += 1
            if j - i >= 20:
                runs.append((i, j))
            i = j
        else:
            i += 1
    if len(runs) < N:
        return t, None
    runs = sorted(sorted(runs, key=lambda r: -(r[1] - r[0]))[:N])
    d_span = (off - on) * 0.01
    dub_pos = [((r[0] + r[1]) / 2 - on) * 0.01 for r in runs]            # dub pause centre, relative to dub onset
    # guard: relative positions must agree (translation kept the sentence order/shape)
    o_pos = [r - 0.2 for r in resumes_rel]                                 # approx. original pause centre
    if any(abs(a / max(slot_dur, 1e-3) - b / max(d_span, 1e-3)) > 0.3 for a, b in zip(o_pos, dub_pos)):
        return t, None
    drift = [r - ((rn[1] * 0.01) - on * 0.01) for r, rn in zip(resumes_rel, runs)]   # orig resume - dub resume
    if max(abs(d) for d in drift) < MIX_ALIGN_MIN_DRIFT:
        return t, None
    # build pieces: piece 0 = [0 .. runs[0] start], piece k = [runs[k-1] end .. runs[k] start], last = [.. end]
    cuts = [(a * 0.01, b * 0.01) for a, b in runs]
    pieces = []
    prev = 0.0
    for (ca, cb) in cuts:
        pieces.append((prev, ca)); prev = cb
    pieces.append((prev, off * 0.01 + MIX_KEEP_TAIL))
    out_parts, cur = [], 0.0       # cur = write head (s) relative to wav start (which sits `lead` before the onset)
    placed = []
    for k, (a, b) in enumerate(pieces):
        seg = t[int(a * sr):int(b * sr)]
        if k == 0:
            start_at = 0.0
        else:
            start_at = max(cur + 0.15, lead + resumes_rel[k - 1])     # resume instants are relative to the onset
        out_parts.append((start_at, seg))
        cur = start_at + len(seg) / sr
        placed.append(start_at)
    total = cur
    if total > max(len(t) / sr, slot_dur + lead) + 0.3:
        return t, None
    if room is not None and total > max(len(t) / sr, room + lead):   # never run into the next slot's start
        return t, None
    new = np.zeros(int(total * sr) + 1, dtype=t.dtype)
    fl = int(0.004 * sr)
    for start_at, seg in out_parts:
        seg = seg.copy()
        if len(seg) > 2 * fl:
            seg[:fl] *= np.linspace(0, 1, fl, dtype=seg.dtype); seg[-fl:] *= np.linspace(1, 0, fl, dtype=seg.dtype)
        i0 = int(start_at * sr)
        new[i0:i0 + len(seg)] += seg[:len(new) - i0]
    return new, dict(n_pauses=N, drift_before=[round(d, 3) for d in drift])
