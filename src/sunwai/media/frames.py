"""Frame and sample ownership of a chunk on the source timeline."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

from sunwai.pipeline.mixing.preservation import _ACTIVITY_HOP


# ---------------------------------------------------------------------------
# Stage 1: fetch
# ---------------------------------------------------------------------------
def _own_from_ts(t0, tb, ticks, r_fps, avg_fps, start, dur):
    """(D): PURE ownership rule on exact rationals. t0 = format start_time (Fraction, may be negative/non-zero), tb = stream
    time_base (Fraction), ticks = DECODED-frame timestamps (ints, presentation order, from ffprobe -show_frames best_effort_timestamp),
    r_fps/avg_fps = reported rates (Fraction), start/dur = chunk interval (Fraction). A frame is owned iff start <= ticks*tb - t0 < start+dur
    (strict, no tolerance). Raises ValueError (caller takes the legacy cut) on duplicate / non-monotonic / missing timestamps, non-CFR
    spacing, or reported fps that disagree with the measured spacing by > 0.5%. Returns (n_owned, delta_seconds_float)."""
    if len(ticks) < 3 or any(t is None for t in ticks):
        raise ValueError("too few / missing decoded timestamps")
    if any(b <= a for a, b in zip(ticks, ticks[1:])):
        raise ValueError("duplicate or non-monotonic decoded timestamps")
    if not (dur > 0 and tb > 0):
        raise ValueError("bad interval / time_base")
    d = [b - a for a, b in zip(ticks, ticks[1:])]
    med = sorted(d)[len(d) // 2]
    if max(abs(x - med) for x in d) * 50 > med:             # more than 2 % spacing jitter = not CFR
        raise ValueError("decoded frame spacing is not constant (VFR)")
    meas = 1 / (med * tb)                                   # measured fps
    for name, f in (("r_frame_rate", r_fps), ("avg_frame_rate", avg_fps)):
        if f is None or f <= 0 or abs(f - meas) * 200 > meas:
            raise ValueError(f"{name} {f} disagrees with the measured frame rate {float(meas):.4f}")
    own = [t for t in ticks if start <= t * tb - t0 < start + dur]
    if not own:
        raise ValueError("no frame inside the chunk")
    return len(own), float(own[0] * tb - t0 - start)


def _owned_frames_by_ts(src, start: float, dur: float):
    """(D): probe `src` (decoded frame timestamps) and apply _own_from_ts. Raises for anything unusable -> legacy cut."""
    import subprocess as _sp, json as _js
    from fractions import Fraction as _F
    if not (math.isfinite(start) and math.isfinite(dur) and start >= 0 and dur > 0):
        raise ValueError("bad chunk interval")
    st, du = _F(repr(float(start))), _F(repr(float(dur)))
    h = _sp.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "format=start_time:stream=time_base,r_frame_rate,avg_frame_rate", "-of", "json", str(src)],
                capture_output=True, text=True, timeout=60)
    j = _js.loads(h.stdout)
    t0 = _F(j["format"]["start_time"])
    s0 = j["streams"][0]
    tb, r_fps, avg_fps = _F(s0["time_base"]), _F(s0["r_frame_rate"]), _F(s0["avg_frame_rate"])
    lo = float(start) + float(t0) - 1.0
    rng = (f"{lo:.6f}%" if lo > 0 else "%") + f"{float(start) + float(dur) + float(t0) + 2.0:.6f}"
    p = _sp.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", rng, "-show_frames",
                 "-show_entries", "frame=best_effort_timestamp", "-of", "json", str(src)], capture_output=True, text=True, timeout=180)
    if p.returncode != 0:
        raise RuntimeError("ffprobe -show_frames failed")
    ticks = [(f.get("best_effort_timestamp") if isinstance(f.get("best_effort_timestamp"), int) else None) for f in _js.loads(p.stdout)["frames"]]
    return _own_from_ts(t0, tb, ticks, r_fps, avg_fps, st, du)


def _frames_to_samples(g: np.ndarray, N: int, sr: int) -> np.ndarray:
    hop = int(_ACTIVITY_HOP * sr)
    xc = (np.arange(len(g)) + 0.5) * hop
    return np.interp(np.arange(N), xc, g).astype(np.float32)


def _chunk_owned_samples(out_dir: Path, N: int, sr: int):
    """returns (samples, exact, reason). exact=True only when the chunk's meta.json yields a plausible owned length
    = round(end*sr) - round(start*sr) (cumulative rounding: chunks tile the source with 0 missing/duplicated samples).
    CHUNK_EXACT_OWN=0 (the single kill switch; legacy alias MUX_FULL_AUDIO=0), missing/invalid/implausible meta -> (N, False, reason)
    and the caller takes the legacy -shortest path."""
    off = ("0", "false", "off", "no")
    if os.environ.get("CHUNK_EXACT_OWN", "1").strip().lower() in off or os.environ.get("MUX_FULL_AUDIO", "1").strip().lower() in off:
        return N, False, "disabled by CHUNK_EXACT_OWN=0"
    try:
        m = json.load(open(out_dir / "meta.json", encoding="utf-8"))
        st, du = float(m["start"]), float(m["dur"])
        if not (math.isfinite(st) and math.isfinite(du) and st >= 0 and du > 0):
            return N, False, f"invalid meta start/dur ({st}, {du})"
        n = int(round((st + du) * sr)) - int(round(st * sr))
        if not (0 < n <= N and N - n <= int(0.25 * sr)):
            return N, False, f"implausible owned length {n} vs {N} samples"
        return n, True, None
    except Exception as e:
        return N, False, f"meta.json unusable ({type(e).__name__}: {e})"


def _audio_track_dur(path: Path) -> Optional[float]:
    try:
        import subprocess as _sp
        o = _sp.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=duration", "-of", "csv=p=0", str(path)],
                    capture_output=True, text=True, timeout=30).stdout.strip()
        return float(o)
    except Exception:
        return None
