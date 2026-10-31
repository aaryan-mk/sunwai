"""Detection and repair ladder for runaway (over-long) syntheses."""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple


log = logging.getLogger(__name__)

from sunwai.media.audio import concat_wav_bytes, trim_wav_bytes, wav_duration


# ---------------------------------------------------------------------------
# Stage 8: TTS + fit
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# runaway ladder: when every TTS retry still returns a runaway (indic-speak keeps generating, ~30.7 s for a 2-9 s slot), the
# pipeline used to KEEP the shortest runaway (garbage in the mix, overflow 15 s). Now: (1) sentence/clause split, each piece synthesised on
# its own (a runaway stays local to its piece); (2) if still too long, trim at slot*1.25 with a fade-out. Never keeps runaway audio.
# Kill switch PIPE_RUNAWAY_LADDER=0.
# ---------------------------------------------------------------------------
def runaway_ladder_on() -> bool:
    return os.environ.get("PIPE_RUNAWAY_LADDER", "1").strip().lower() not in ("0", "false", "no", "off", "n", "f")


def _runaway_gps() -> float:
    try:
        v = float(os.environ.get("PIPE_RUNAWAY_GPS", "8"))
        return v if 1.0 <= v <= 50.0 else 8.0
    except Exception:
        return 8.0


def is_runaway(d: float, slot: float, text: Optional[str] = None) -> bool:
    """Runaway TTS (indic-speak keeps generating). True when (a) the audio reaches the ~30.7 s server cap and is far beyond the slot, or
    (b) it is >4 s longer than the slot AND (text known) more than 2x what a SLOW reader (PIPE_RUNAWAY_GPS=8 chars/s) needs for that text
    -- a long text read at a normal pace is not a runaway and is left to the fit/refit path -- or (text unknown) more than 2.5x the slot."""
    if d >= 29.5 and d > slot * 1.5:
        return True
    if d <= slot + 4.0:
        return False
    if text:
        import re as _re
        g = len(_re.sub(r"\s+", "", text))
        return bool(g) and d > 2.0 * g / _runaway_gps()
    return d / max(slot, 1e-6) > 2.5


def runaway_split_pieces(text: str, n_max: int = 3, min_g: int = 12) -> List[str]:
    """Split a target text at clause/sentence boundaries into <= n_max balanced pieces (each >= min_g characters); [] if it cannot be split."""
    import re as _re
    parts = [p.strip() for p in _re.split(r"(?<=[.!?\u2026\u2014;:\u0964\u061f,])\s+|\s*\u2014\s*", text or "") if p and p.strip()]
    if len(parts) < 2:
        return []
    total = sum(len(p) for p in parts)
    target = total / min(n_max, len(parts))
    out, cur = [], ""
    for p in parts:
        cur = (cur + " " + p).strip() if cur else p
        if len(cur) >= max(min_g, target * 0.9) and len(out) < n_max - 1:
            out.append(cur); cur = ""
    if cur:
        if out and len(cur) < min_g:
            out[-1] = out[-1] + " " + cur
        else:
            out.append(cur)
    return out if len(out) >= 2 else []


def runaway_ladder(text: str, slot: float, best_wb: bytes, best_d: float, synth_piece, log=None) -> Tuple[bytes, float, str]:
    """-> (wav bytes, duration, how). synth_piece(piece_text, span_s) -> wav bytes (may raise). how: 'split' | 'trim' | 'none'."""
    say = (log.warning if log else (lambda *a, **k: None))
    pieces = runaway_split_pieces(text)
    if pieces:
        try:
            tot = sum(len(p) for p in pieces)
            wbs = [synth_piece(p, slot * len(p) / tot) for p in pieces]
            wb = concat_wav_bytes(wbs)
            d = wav_duration(wb)
            if not is_runaway(d, slot, text) and d < best_d * 0.8:
                return wb, d, "split"
            say("runaway ladder: split result still %.1fs for %.1fs slot", d, slot)
        except Exception as e:
            say("runaway ladder: split failed (%s)", e)
    cap = slot * 1.25 + 0.05
    try:
        wb = trim_wav_bytes(best_wb, cap)
        return wb, min(best_d, cap), "trim"
    except Exception as e:
        say("runaway ladder: trim failed (%s)", e)
        return best_wb, best_d, "none"
