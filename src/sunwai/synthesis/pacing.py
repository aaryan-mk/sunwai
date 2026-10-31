"""TTS latency tracking and slow-service detection."""

from __future__ import annotations

import logging
import threading as _threading
from typing import List


log = logging.getLogger(__name__)

from sunwai.pipeline.support.env import _envf, _flag


_TTS_LAT: List[float] = []                    # durations (s) of this chunk's recent TTS calls (one process per chunk)


_TTS_LAT_LOCK = _threading.Lock()


def _record_lat(dt: float, wb=None) -> None:
    """latency bookkeeping: can never invalidate successful audio. Stores the real-time factor (call seconds per second of audio
    produced), so long lines are not mistaken for a loaded server."""
    try:
        aud = None
        try:
            import io as _io, wave as _wv
            _w = _wv.open(_io.BytesIO(wb))
            aud = _w.getnframes() / float(_w.getframerate())
        except Exception:
            aud = None
        if not aud or aud <= 0:
            return                                           # unknown output length: no sample (never guess)
        with _TTS_LAT_LOCK:
            _TTS_LAT.append(float(dt) / max(0.3, aud))
            del _TTS_LAT[:-8]
    except BaseException as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        try:
            log.warning("tts latency record skipped: %s", e)
        except Exception:
            pass


def _tts_slow_now() -> bool:
    """True when the rolling median real-time factor of the last 8 TTS calls exceeds TTS_SENT_SLOW_RTF (default 1.5 s of call per
    s of audio): splitting multiplies calls under load, so a loaded server gets legacy only."""
    with _TTS_LAT_LOCK:
        v = sorted(_TTS_LAT)
    return bool(v) and v[len(v) // 2] > _envf("TTS_SENT_SLOW_RTF", 1.5, 0.2, 100.0)


def _tts_slow_min() -> float:
    """lowest atempo for a slow-down (TTS shorter than its slot); 0.9 = natural-sounding limit. TTS_SLOW=0 disables."""
    return _envf("TTS_SLOW_MIN", 0.9, 0.9, 1.0) if _flag("TTS_SLOW") else 1.0
