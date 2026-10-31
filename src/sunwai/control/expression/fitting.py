"""Audio fitting: synthesise a take, trim it, speed it up a little if it overruns, and pad short takes to the slot."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf

TEMPO_LO, TEMPO_HI = 0.92, 1.08
OVERRUN_TOLERANCE_S = 1e-3

TtsFn = Callable[[str, str, str, str | None], bytes]
TrimFn = Callable[[Path, Path], tuple[Path, float]]
TempoFn = Callable[[Path, Path, float], None]


class Take(NamedTuple):
    """One fitted synthesis attempt."""

    path: Path
    raw_s: float
    tempo: float
    fitted_s: float
    overruns: bool


def tempo_for(duration_s: float, slot_s: float) -> float:
    """Return the atempo factor for a take: 1.0 unless it overruns the slot, then clamped to the supported range."""
    ratio = duration_s / slot_s
    return 1.0 if ratio <= 1.0 else min(max(ratio, TEMPO_LO), TEMPO_HI)


def wav_duration(path: str | Path) -> float:
    """Return the duration of an audio file in seconds."""
    info = sf.info(str(path))
    return float(info.frames / float(info.samplerate))


def ffmpeg_tempo(src: Path, dst: Path, tempo: float) -> None:
    """Time-stretch ``src`` into ``dst`` (mono, 24 kHz) with ffmpeg ``atempo``; the call is time-bounded."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-filter:a", f"atempo={tempo:.6f}", "-ac", "1", "-ar", "24000", str(dst)],
        check=True,
        timeout=float(os.environ.get("EXPR_FFMPEG_TIMEOUT", "120")),
    )


def pad_to(path: str | Path, duration_s: float) -> None:
    """Pad an audio file with trailing silence up to ``duration_s`` (never shortens it)."""
    samples, rate = sf.read(str(path), dtype="float32")
    wanted = int(round(duration_s * rate))
    if len(samples) < wanted:
        sf.write(str(path), np.concatenate([samples, np.zeros(wanted - len(samples), dtype="float32")]), rate, subtype="PCM_16")


def synth_fit(
    text: str,
    lang: str,
    style: str | None,
    voice: str,
    cand_dir: Path,
    tag: str,
    *,
    tts: TtsFn,
    trim: TrimFn,
    tempo_fn: TempoFn,
    allowed_s: float,
    slot_s: float,
) -> Take:
    """Synthesise ``text``, trim it and tempo-fit it into ``cand_dir/<tag>.wav``.

    Only overruns are sped up; underruns are padded later by the caller.

    Args:
        text: Spoken text.
        lang: Language code.
        style: Style tag or ``None``.
        voice: Voice name.
        cand_dir: Directory for the candidate files.
        tag: File-name stem of this attempt.
        tts: Synthesiser returning WAV bytes.
        trim: Silence trimmer returning ``(path, duration_s)``.
        tempo_fn: Time stretcher.
        allowed_s: Longest acceptable fitted duration.
        slot_s: Duration of the line's slot.
    """
    raw = cand_dir / f"{tag}_raw.wav"
    raw.write_bytes(tts(text, lang, voice, style))
    fit_src, duration = trim(raw, cand_dir / f"{tag}_trim.wav")
    tempo = tempo_for(duration, slot_s)
    out = cand_dir / f"{tag}.wav"
    tempo_fn(fit_src, out, tempo)
    fitted = wav_duration(out)
    return Take(out, duration, tempo, fitted, fitted > allowed_s + OVERRUN_TOLERANCE_S)
