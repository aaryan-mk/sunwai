"""Sentence and pause based splitting of long lines for synthesis, plus pause stretching."""

from __future__ import annotations

import json
import logging
import threading as _threading
from pathlib import Path
from typing import Any, Dict

import numpy as np

log = logging.getLogger(__name__)

from sunwai.pipeline.mixing.levels import _frame_db, _speech_bounds
from sunwai.pipeline.support.env import _envf, _flag


_SPLIT_SEM = None


_SPLIT_SEM_LOCK = _threading.Lock()


def _split_sem():
    global _SPLIT_SEM
    with _SPLIT_SEM_LOCK:
        if _SPLIT_SEM is None:
            _SPLIT_SEM = _threading.BoundedSemaphore(int(_envf("TTS_SENT_MAX_CONC", 3, 1, 8)))
        return _SPLIT_SEM


def _src_subspans(out_dir: Path, seg: Dict[str, Any], min_gap: float = None, max_pieces: int = None):
    """Source speech sub-spans of one slot, chunk time: VAD intervals clipped to the slot and joined unless separated by a
    silence >= min_gap. Returns None when the slot has no internal pause that long (or on any problem)."""
    min_gap = min_gap if min_gap is not None else _envf("TTS_SENT_MIN_GAP", 0.35, 0.2, 2.0)
    max_pieces = max_pieces if max_pieces is not None else int(_envf("TTS_SENT_MAX_PIECES", 4, 2, 8))
    try:
        iv = json.load(open(Path(out_dir) / "vad.json", encoding="utf-8"))
        iv = [((v["start"], v["end"]) if isinstance(v, dict) else (v[0], v[1])) for v in iv]
        s, e = float(seg["start"]), float(seg["end"])
        iv = sorted((max(a, s), min(b, e)) for a, b in iv if b > s + 0.01 and a < e - 0.01)
        iv = [x for x in iv if x[1] - x[0] > 0.02]
        if not iv:
            return None
        spans = [list(iv[0])]
        for a, b in iv[1:]:
            if a - spans[-1][1] >= min_gap:
                spans.append([a, b])
            else:
                spans[-1][1] = max(spans[-1][1], b)
        while len(spans) > max_pieces:                       # keep the longest pauses
            k = min(range(len(spans) - 1), key=lambda i: spans[i + 1][0] - spans[i][1])
            spans[k][1] = spans[k + 1][1]
            del spans[k + 1]
        return [(a, b) for a, b in spans] if len(spans) >= 2 else None
    except Exception:
        return None


_SPLIT_STRONG = ".?!;:।॥؟۔…"


_SPLIT_WEAK = ",،、—–"


def _g_count(txt: str) -> int:
    try:
        from sunwai.pipeline.fitting.fit_stage import graphemes
        return graphemes(txt)
    except Exception:
        return sum(1 for ch in txt if ch.isalnum())


def _split_target(text: str, spans, min_g: int = None, word_split: bool = None):
    """Split the translated line into len(spans) pieces so that piece k is spoken over source sub-span k. Boundaries are
    taken at sentence punctuation (then commas, then - only if word_split - any word boundary) nearest to the source's
    cumulative speech-time fraction. A boundary that cannot be placed within tolerance, or that leaves a piece with fewer
    than min_g graphemes, is dropped (neighbouring sub-spans merge). Returns [(piece_text, t0, t1)] or None (<2 pieces)."""
    import re as _re
    min_g = min_g if min_g is not None else int(_envf("TTS_SENT_MIN_G", 8, 2, 40))
    word_split = _flag("TTS_SENT_WORDSPLIT", "1") if word_split is None else word_split
    text = " ".join((text or "").split())
    K = len(spans)
    if K < 2 or not text:
        return None
    dur = [b - a for a, b in spans]
    tot = sum(dur) or 1e-6
    fr = [sum(dur[:i + 1]) / tot for i in range(K - 1)]            # cumulative speech-time fraction at each boundary
    toks = list(_re.finditer(r"\S+", text))
    G = max(1, _g_count(text))
    cands = []                                                      # (char_end, graphemes_before_frac, strength)
    for m in toks[:-1]:
        w = m.group().rstrip("\"'”’)]}")
        st = 2 if (w and w[-1] in _SPLIT_STRONG) else (1 if (w and w[-1] in _SPLIT_WEAK) else 0)
        if st == 0 and not word_split:
            continue
        cands.append((m.end(), _g_count(text[:m.end()]) / G, st))
    cuts, last = [], 0.0                                            # chosen (char_end, boundary index)
    for j, f in enumerate(fr):
        best = None
        for ce, cf, st in cands:
            if cf <= last + 0.03 or (cuts and ce <= cuts[-1][0]):
                continue
            tol = 0.12 if st == 0 else 0.22
            if abs(cf - f) > tol:
                continue
            cost = abs(cf - f) + (0.0, 0.02, 0.0)[st] + (0.06 if st == 0 else 0.0)
            if best is None or cost < best[0]:
                best = (cost, ce, cf)
        if best:
            cuts.append((best[1], j))
            last = best[2]
    if not cuts:
        return None
    # pieces of text; the sub-span list is merged across every dropped boundary
    kept = {j for _, j in cuts}
    groups, cur = [], [0]
    for i in range(1, K):
        if (i - 1) in kept:
            groups.append(cur); cur = [i]
        else:
            cur.append(i)
    groups.append(cur)
    ces = [0] + [ce for ce, _ in cuts] + [len(text)]
    pieces = [[text[ces[i]:ces[i + 1]].strip(), spans[g[0]][0], spans[g[-1]][1]] for i, g in enumerate(groups)]
    # drop pieces that are too small to synthesise well: merge into a neighbour
    i = 0
    while len(pieces) > 1 and i < len(pieces):
        if _g_count(pieces[i][0]) < min_g:
            j = i - 1 if i > 0 else i + 1
            a, b = (j, i) if j < i else (i, j)
            pieces[a] = [(pieces[a][0] + " " + pieces[b][0]).strip(), pieces[a][1], pieces[b][2]]
            del pieces[b]
            i = 0
        else:
            i += 1
    if len(pieces) < 2 or any(not p[0] for p in pieces):
        return None
    return [(p[0], p[1], p[2]) for p in pieces]


def _stretch_pauses(y: np.ndarray, sr: int, leftover: float, max_each: float = None, frac: float = None):
    """Spend part of `leftover` seconds of slack by lengthening the TTS's own internal pauses (>= 0.12 s of silence between
    speech), proportionally to their length, at most max_each per pause. Returns (new_y, added_s)."""
    max_each = max_each if max_each is not None else _envf("TTS_SLACK_MAX_PAUSE", 0.35, 0.05, 1.5)
    frac = frac if frac is not None else _envf("TTS_SLACK_FRAC", 0.7, 0.1, 1.0)
    db = _frame_db(y, sr)
    on, off, act = _speech_bounds(db, 30.0)
    if on is None:
        return y, 0.0
    runs, i = [], on
    while i < off:
        if not act[i]:
            j = i
            while j < off and not act[j]:
                j += 1
            if j - i >= 12:
                runs.append((i, j))
            i = j
        else:
            i += 1
    if not runs or leftover <= 0.03:
        return y, 0.0
    budget = leftover * frac
    lens = np.array([(b - a) * 0.01 for a, b in runs])
    add = np.minimum(max_each, budget * lens / lens.sum())
    # water-fill what the caps left over
    rest = budget - float(add.sum())
    for _ in range(3):
        room = max_each - add
        if rest <= 1e-3 or room.sum() <= 1e-6:
            break
        d = np.minimum(room, rest * room / room.sum()); add += d; rest -= float(d.sum())
    parts, prev = [], 0
    for (a, b), x in zip(runs, add):
        mid = int(((a + b) / 2) * 0.01 * sr)
        parts.append(y[prev:mid]); parts.append(np.zeros(int(x * sr), dtype=y.dtype)); prev = mid
    parts.append(y[prev:])
    return np.concatenate(parts), float(add.sum())


_MISSING = object()


class _SplitRestoreFailed(Exception):
    pass


_HOP = 0.01          # _frame_db default hop (s); the silence thresholds below are in these frames


def _whole_pause_drift(wav_path, spans, s0: float):
    """max |dub pause centre - source pause centre| (s) of the already-fitted whole-slot audio, pauses matched in order
    (ANY internal silence >= 0.12 s, matched in order to the N source gaps so that the largest offset is minimal). None when it cannot be told (unreadable); inf when the audio has
    fewer pauses than the source (cannot be aligned without splitting)."""
    try:
        import soundfile as _sf
        y, sr = _sf.read(str(wav_path), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        db = _frame_db(y, sr)
        on, off, act = _speech_bounds(db, 30.0)
        if on is None:
            return None
        runs, i = [], on
        while i < off:
            if not act[i]:
                j = i
                while j < off and not act[j]:
                    j += 1
                if j - i >= int(round(0.12 / _HOP)):
                    runs.append((i, j))
                i = j
            else:
                i += 1
        src = [((spans[k][1] + spans[k + 1][0]) / 2 - s0) for k in range(len(spans) - 1)]
        if len(runs) < len(src):
            return float("inf")
        cen = [((a + b) / 2) * _HOP for a, b in runs]
        # order-constrained assignment of the N source gaps to N of the dub's silences minimising the LARGEST offset (bottleneck DP):
        # extra breath/comma silences cannot displace the genuine pause, and the result is the best achievable max drift
        N, R = len(src), len(cen)
        INF = float("inf")
        best = [[INF] * (R + 1) for _ in range(N + 1)]
        for r in range(R + 1):
            best[0][r] = 0.0
        for n in range(1, N + 1):
            for r in range(n, R + 1):
                best[n][r] = min(best[n][r - 1], max(best[n - 1][r - 1], abs(cen[r - 1] - src[n - 1])))
        return best[N][R]
    except Exception:
        return None
