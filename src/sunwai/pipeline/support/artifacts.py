"""Fingerprints and freshness checks for cached synthesis artifacts."""

from __future__ import annotations

import hashlib
import logging
import os
import threading as _threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

from sunwai.clients.service_endpoints import TTS_MODEL


def tts_fingerprint(text: str, voice: str, lang: str, style=None) -> str:
    model = TTS_MODEL
    parts = [text, voice, lang, model] + ([style] if style else [])   # no style => fingerprint unchanged
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()


def write_raw_tts(raw_path: Path, wav_bytes: bytes, fp: str) -> None:
    """raw wav first (atomic), fingerprint sidecar second: a crash in between leaves a raw with no fp = regenerated."""
    tmp = raw_path.with_name(f".{raw_path.stem}.tmp{os.getpid()}_{_threading.get_ident()}.wav")
    tmp.write_bytes(wav_bytes)
    try:
        os.remove(raw_path.with_name(raw_path.stem + ".fp"))   # invalidate the old fingerprint BEFORE the audio changes
    except OSError:
        pass
    os.replace(tmp, raw_path)
    (raw_path.with_name(raw_path.stem + ".fp")).write_text(fp)


def raw_tts_fresh(raw_path: Path, fp: str) -> bool:
    try:
        return raw_path.exists() and raw_path.with_name(raw_path.stem + ".fp").read_text().strip() == fp
    except Exception:
        return False


def mix_mask(spans, N: int, sr: int = 44100) -> np.ndarray:
    """spans = [(start_sample, end_sample)] of regions where the original vocals are replaced (bed + TTS).
    Returns the Hann-smoothed 0..1 mask. The caller passes the END of the TTS audio when it spills past the slot
    (borrowed following silence) so spilled speech is NOT attenuated."""
    from scipy.signal.windows import hann
    cf = int(0.08 * sr)
    mask = np.zeros(N, dtype=np.float32)
    for a, b in spans:
        mask[max(0, a):min(N, b)] = 1.0
    win = hann(2 * cf + 1, sym=False)
    kernel = np.concatenate([win[:cf], win[cf:]])
    return np.clip(np.convolve(mask, kernel, mode="same") / kernel.sum(), 0.0, 1.0)
