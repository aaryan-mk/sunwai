"""Single-line re-synthesis after an edit."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import librosa

log = logging.getLogger(__name__)

from sunwai.clients.service_calls import bodhan_tts, load_keys
from sunwai.media.ffmpeg import run_ffmpeg
from sunwai.pipeline.mixing.levels import _pretrim_wav
from sunwai.pipeline.stages.mix import stage_mix
from sunwai.pipeline.support.artifacts import tts_fingerprint, write_raw_tts
from sunwai.pipeline.support.concurrency import _fault, thread_session
from sunwai.pipeline.support.expression_flags import _expr_style
from sunwai.pipeline.support.storage import atomic_write_json
from sunwai.pipeline.voices import pick_voice, voice_require_speakers


# ---------------------------------------------------------------------------
# Resynth
# ---------------------------------------------------------------------------
def do_resynth(args, out_dir: Path):
    """Re-synthesise one edited line (``args.resynth``) with ``args.text`` and remix the chunk.

    Only the edited segment and the downstream mix are recomputed.
    """
    seg_path = out_dir / "segments.json"
    if not seg_path.exists():
        raise RuntimeError("segments.json not found; run full pipeline first")
    with open(seg_path, encoding="utf-8") as f:
        segments = json.load(f)
    seg = next((s for s in segments if s["id"] == args.resynth), None)
    if seg is None:
        raise RuntimeError(f"segment id {args.resynth} not found")

    # edit history is owned by the server (edit pushes, undo pops); appending here made undo a no-op
    seg["tgt_text"] = args.text
    seg["edited"] = True
    seg.pop("sent_pieces", None)       # the edit is whole-slot audio; drop stale placement metadata
    seg.pop("slack_added", None)

    keys = load_keys()
    session = thread_session()
    tts_key = keys["tts"]
    tts_dir = out_dir / "tts"
    tts_dir.mkdir(exist_ok=True)
    raw_path = tts_dir / f"{args.resynth}_raw.wav"
    out_path = tts_dir / f"{args.resynth}.wav"
    voice_require_speakers([seg])
    try:
        voice = pick_voice(args.lang, seg["gender"], seg["speaker"])
        _fault("tts", args.resynth)
        wav_bytes = bodhan_tts(session, tts_key, args.text, args.lang, voice, style=_expr_style(seg))
        write_raw_tts(raw_path, wav_bytes, tts_fingerprint(args.text, voice, args.lang, _expr_style(seg)))
    except Exception as e:
        log.error("Resynth TTS error: %s", e)
        raise
    # refit
    fit_src, tts_dur = _pretrim_wav(raw_path, tts_dir / f"{args.resynth}_trim.wav")
    slot_dur = max(seg["end"] - seg["start"], 1e-6)
    ratio = tts_dur / slot_dur
    tempo = min(ratio, 1.25) if ratio > 1.0 else 1.0
    run_ffmpeg([
        "-i", str(fit_src),
        "-filter:a", f"atempo={tempo:.6f}",
        "-ac", "1", "-ar", "24000",
        str(out_path),
    ])
    if fit_src is not raw_path:
        try:
            os.remove(fit_src)
        except OSError:
            pass
    fitted_dur = librosa.get_duration(path=str(out_path))
    nxt = next((s for s in segments if s["id"] == seg["id"] + 1), None)
    allowed = slot_dur
    if nxt is not None:
        allowed = max(slot_dur, nxt["start"] - seg["start"])
    if fitted_dur > allowed + 1e-3:
        seg["overflow"] = True
        seg["overflow_s"] = fitted_dur - allowed
        run_ffmpeg([
            "-i", str(out_path), "-t", f"{allowed}",
            "-af", f"afade=t=out:st={max(0,allowed-0.05)}:d=0.05",
            "-ac", "1", "-ar", "24000", str(out_path),
        ])
    else:
        seg["overflow"] = False
        seg["overflow_s"] = 0.0
    seg["ratio"] = ratio
    seg["tempo"] = tempo
    seg["voice"] = voice
    # a successful re-voicing clears any earlier fallback flag on this segment
    if seg.get("fallback"):
        for k in ("fallback", "fallback_reason", "tts_error", "mt_error", "asr_error"):
            seg.pop(k, None)
        seg["error"] = None

    atomic_write_json(seg_path, segments, indent=2)

    # redo mix (bed cached)
    out_mp4 = stage_mix(args, out_dir, segments)
    print(out_mp4)
    return out_mp4
