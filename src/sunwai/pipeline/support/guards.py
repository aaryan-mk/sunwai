"""Media validity checks and the fallback-speech budget."""

from __future__ import annotations

import json
import logging
import os
from typing import Tuple


log = logging.getLogger(__name__)




def media_ok(path, need_video=False, need_audio=True, min_dur=0.05) -> bool:
    """ffprobe validation: file exists, parses, has the required streams and duration >= min_dur."""
    import subprocess
    try:
        if not os.path.exists(path) or os.path.getsize(path) < 64:
            return False
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type:format=duration",
                            "-of", "json", str(path)], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return False
        js = json.loads(r.stdout or "{}")
        kinds = {x.get("codec_type") for x in js.get("streams", [])}
        if need_video and "video" not in kinds:
            return False
        if need_audio and "audio" not in kinds:
            return False
        return float(js.get("format", {}).get("duration") or 0.0) >= min_dur
    except Exception:
        return False


def failed_speech_frac(segments) -> Tuple[float, float, float]:
    """(failed_seconds, total_speech_seconds, fraction) over segments flagged fallback."""
    tot = sum(max(0.0, s["end"] - s["start"]) for s in segments)
    bad = sum(max(0.0, s["end"] - s["start"]) for s in segments
              if s.get("fallback") and s.get("fallback_reason") in ("asr", "mt", "tts"))
    return bad, tot, (bad / tot if tot > 0 else 0.0)


def enforce_fallback_budget(segments, stage: str) -> None:
    """Fail the chunk when failed speech exceeds FALLBACK_MAX_FRAC (default 0.20) of the chunk's speech."""
    bad, tot, frac = failed_speech_frac(segments)
    if bad > 0:
        log.warning("FALLBACK after %s: %d seg(s) %.1fs of %.1fs speech (%.0f%%) keep the ORIGINAL audio: %s", stage,
                    sum(1 for x in segments if x.get("fallback_reason") in ("asr", "mt", "tts")), bad, tot, 100 * frac,
                    [(x["id"], x.get("fallback_reason")) for x in segments if x.get("fallback_reason") in ("asr", "mt", "tts")])
    if frac > float(os.environ.get("FALLBACK_MAX_FRAC", "0.20")):
        raise RuntimeError(f"chunk failed: {frac:.0%} of speech ({bad:.1f}s/{tot:.1f}s) failed at {stage} (limit "
                           f"{float(os.environ.get('FALLBACK_MAX_FRAC', '0.20')):.0%})")


def mark_fallback(seg, reason: str, err) -> None:
    """Segment keeps the original audio: no dubbed speech, original vocals NOT masked, never speaks untranslated source."""
    seg["fallback"] = True
    seg["fallback_reason"] = reason
    seg["error"] = seg.get("error") or f"{reason}: {err}"
    seg[f"{reason}_error"] = str(err)
