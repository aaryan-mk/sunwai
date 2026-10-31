"""Speech trimming, peak limiting and level helpers for the mix."""

from __future__ import annotations

import logging
import math
import os

import librosa
import numpy as np

log = logging.getLogger(__name__)

from sunwai.pipeline.support.env import _envf


# ---------------------------------------------------------------------------
# speech-onset alignment helpers (trim TTS lead/trail silence, per-pause re-layout)
# ---------------------------------------------------------------------------
def _frame_db(y: np.ndarray, sr: int, hop_s: float = 0.01) -> np.ndarray:
    hop = max(1, int(hop_s * sr))
    n = len(y) // hop
    if n == 0:
        return np.zeros(0)
    f = y[:n * hop].reshape(n, hop)
    return 10.0 * np.log10(np.mean(f * f, axis=1) + 1e-10)


def _speech_bounds(db: np.ndarray, rel: float, need: int = 4):
    """(first, last_exclusive, active_mask) frame indices of sustained speech; threshold = p95 - rel dB, floor -70 dB."""
    if len(db) < need:
        return None, None, None
    act = db > max(float(np.percentile(db, 95)) - rel, -70.0)
    on = off = None
    for i in range(len(act) - need + 1):
        if act[i:i + need].all():
            on = i
            break
    for i in range(len(act), need - 1, -1):
        if act[i - need:i].all():
            off = i
            break
    if on is None or off is None or off <= on:
        return None, None, None
    return on, off, act


MIX_TRIM_REL = _envf("MIX_TRIM_REL", 35, 10.0, 80.0)      # conservative: only cuts audio >35 dB below the p95 level


MIX_KEEP_LEAD = _envf("MIX_KEEP_LEAD", 0.03, 0.0, 0.5)  # s of audio kept before the detected onset


MIX_KEEP_TAIL = _envf("MIX_KEEP_TAIL", 0.05, 0.0, 1.0)


def _pretrim_wav(raw_path, dst):
    """trim lead/trail silence BEFORE duration fitting (the mix trims anyway, so fitting/truncating the untrimmed
    audio over-speeds or clips the last word). Returns (path_to_fit, duration_s). The raw TTS cache is never modified; when
    the trim is not clearly safe (kill switch FIT_TRIM=0 / MIX_TRIM=0, no sustained speech found, nothing to cut, or more than
    half of the audio would be cut) the untrimmed raw file is used unchanged."""
    raw_dur = librosa.get_duration(path=str(raw_path))
    if os.environ.get("FIT_TRIM", "1") != "1" or os.environ.get("MIX_TRIM", "1") != "1":
        return raw_path, raw_dur
    try:
        t, sr0 = librosa.load(str(raw_path), sr=None, mono=True, dtype="float32")
        if len(t) == 0:
            return raw_path, raw_dur
        tt, _lead = _trim_tts(t, sr0)
        if len(tt) >= len(t) - int(0.01 * sr0) or len(tt) < 0.5 * len(t):
            return raw_path, raw_dur
        import soundfile as _sf
        _sf.write(str(dst), tt, sr0, subtype="PCM_16")
        return dst, len(tt) / sr0
    except Exception as _e:
        log.warning("pretrim skipped (%s): %s", raw_path, _e)
        return raw_path, raw_dur


def _peak_limit(x: np.ndarray, ceiling: float, sr: int, look_ms: float, rel_ms: float) -> np.ndarray:
    """sample-accurate look-ahead TRUE-peak limiter (x = T x C; 4x oversampled detection). Offline, so the look-ahead needs no latency: the gain
    at sample n is a centred +-look window minimum of the required gain, smoothed over the same window (so it is always
    <= the required gain at n, i.e. |y| <= ceiling), followed by a one-pole release (gain recovers ~90% in rel_ms).
    Chunks whose peak is already <= ceiling are returned unchanged; gain reduction is local to the overs."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    p = np.max(np.abs(x), axis=1)
    if not (float(p.max()) * 1.5 > ceiling):          # cannot reach the ceiling even with a 3.5 dB inter-sample over
        return x
    # inter-sample (true) peak: 4x oversampled |x|, max-pooled back onto the original sample grid
    from scipy.signal import resample_poly
    p = np.max(np.abs(resample_poly(x.astype(np.float32, copy=False), 4, 1, axis=0)), axis=1)[:4 * len(x)]
    p = p.reshape(len(x), 4).max(axis=1)
    if not (float(p.max()) > ceiling):
        return x
    g = np.minimum(1.0, ceiling / np.maximum(p, 1e-12)).astype(np.float64)
    L = max(1, int(look_ms * 1e-3 * sr))
    ga = uniform_filter1d(minimum_filter1d(g, size=2 * L + 1, mode="nearest"), size=2 * L + 1, mode="nearest")
    ga = np.minimum(ga, g)
    a = math.exp(-math.log(10.0) / max(1.0, rel_ms * 1e-3 * sr))
    n_tot = len(ga)
    act = np.flatnonzero(ga < 1.0)
    r = np.ones(n_tot, dtype=np.float64)
    gl = ga.tolist()
    prev = 1.0
    n = int(act[0])
    while n < n_tot:
        v = 1.0 - (1.0 - prev) * a
        gv = gl[n]
        if gv < v:
            v = gv
        r[n] = v
        prev = v
        n += 1
        if 1.0 - prev < 1e-6 and n < n_tot and gl[n] >= 1.0:
            k = int(np.searchsorted(act, n))
            if k >= len(act):
                break
            n = int(act[k])
            prev = 1.0
    y = x * r[:, None].astype(x.dtype)
    return np.clip(y, -ceiling, ceiling).astype(x.dtype)


def _trim_tts(t: np.ndarray, sr: int):
    """Trim leading/trailing silence of one TTS wav. Returns (trimmed, lead_kept_s). No-op when nothing meaningful to cut."""
    db = _frame_db(t, sr)
    on, off, _ = _speech_bounds(db, MIX_TRIM_REL)
    if on is None:
        return t, 0.0
    a = max(0, int((on * 0.01 - MIX_KEEP_LEAD) * sr))
    b = min(len(t), int((off * 0.01 + MIX_KEEP_TAIL) * sr))
    if a < int(0.01 * sr) and b > len(t) - int(0.01 * sr):
        return t, MIX_KEEP_LEAD if on * 0.01 >= MIX_KEEP_LEAD else on * 0.01
    out = t[a:b].copy()
    fl = min(int(0.005 * sr), len(out) // 4)           # 5 ms edge fades: no click at the cut
    if fl > 0:
        if a > 0:
            out[:fl] *= np.linspace(0.0, 1.0, fl, dtype=out.dtype)
        if b < len(t):
            out[-fl:] *= np.linspace(1.0, 0.0, fl, dtype=out.dtype)
    return out, (on * 0.01 - a / sr)


MIX_LIMITER = os.environ.get("MIX_LIMITER", "1") == "1"            # (0 = old whole-chunk peak scaling)


MIX_LIM_LOOKAHEAD_MS = _envf("MIX_LIM_LOOKAHEAD_MS", 5.0, 0.5, 20.0)


MIX_LIM_RELEASE_MS = _envf("MIX_LIM_RELEASE_MS", 80.0, 10.0, 500.0)   # time to ~90% gain recovery


MIX_LEVEL_FIX = os.environ.get("MIX_LEVEL_FIX", "1") == "1"        # (0 = old loudness match)
