"""WAV reading, writing, validation and byte-level helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import librosa
import numpy as np

log = logging.getLogger(__name__)




def load_wav_mono(path: Path, sr: int = 16000) -> np.ndarray:
    """Load ``path`` as mono float32 at ``sr`` Hz."""
    y, _ = librosa.load(str(path), sr=sr, mono=True, dtype=np.float32)
    return y


def load_wav_stereo(path: Path, sr: int = 44100) -> np.ndarray:
    """Load ``path`` as float32 with shape (channels, samples) at ``sr`` Hz."""
    y, _ = librosa.load(str(path), sr=sr, mono=False, dtype=np.float32)
    if y.ndim == 1:
        y = np.stack([y, y])
    return y.T  # T x 2


def save_wav_mono(path: Path, y: np.ndarray, sr: int):
    """Write mono samples ``y`` to ``path`` at ``sr`` Hz."""
    import soundfile as sf
    sf.write(str(path), y, sr)


def save_wav_stereo(path: Path, y: np.ndarray, sr: int):
    """Write (channels, samples) audio ``y`` to ``path`` at ``sr`` Hz."""
    import soundfile as sf
    if y.ndim == 1:
        y = np.stack([y, y], axis=1)
    sf.write(str(path), y, sr)


def crop_to_seconds(path: Path, out_path: Path, start: float, dur: float,
                    sr: int = 16000) -> np.ndarray:
    """Crop wav file to [start, start+dur] seconds at given sample rate."""
    y = load_wav_mono(path, sr=sr)
    a = int(round(start * sr))
    b = int(round((start + dur) * sr))
    b = min(b, len(y))
    a = max(0, min(a, b))
    return y[a:b]


# ---------------------------------------------------------------------------
# Stage 2: audio extraction
# ---------------------------------------------------------------------------
def _wav_ok(path) -> bool:
    try:
        import soundfile as _sf
        i = _sf.info(str(path))
        return i.frames > 0 and i.samplerate > 0
    except Exception:
        return False


def _wav_pcm(wb: bytes):
    import io as _io, wave as _wave
    w = _wave.open(_io.BytesIO(wb))
    return w.getframerate(), w.getnchannels(), w.getsampwidth(), w.readframes(w.getnframes())


def _wav_bytes(sr: int, ch: int, sw: int, pcm: bytes) -> bytes:
    import io as _io, wave as _wave
    b = _io.BytesIO()
    w = _wave.open(b, "wb"); w.setnchannels(ch); w.setsampwidth(sw); w.setframerate(sr); w.writeframes(pcm); w.close()
    return b.getvalue()


def wav_duration(wb: bytes) -> float:
    """Return the duration in seconds of WAV bytes ``wb``."""
    sr, ch, sw, pcm = _wav_pcm(wb)
    return len(pcm) / (ch * sw) / sr


def concat_wav_bytes(wbs: List[bytes], gap_s: float = 0.12) -> bytes:
    """Concatenate WAV byte strings with ``gap_s`` seconds of silence between them.

    All inputs must share sample rate, channel count and sample width.
    """
    sr, ch, sw, first = _wav_pcm(wbs[0])
    gap = b"\x00" * (int(gap_s * sr) * ch * sw)
    parts = [first]
    for wb in wbs[1:]:
        sr2, ch2, sw2, pcm = _wav_pcm(wb)
        if (sr2, ch2, sw2) != (sr, ch, sw):
            raise ValueError("wav format mismatch")
        parts += [gap, pcm]
    return _wav_bytes(sr, ch, sw, b"".join(parts))


def trim_wav_bytes(wb: bytes, max_s: float, fade_s: float = 0.12) -> bytes:
    """Cut at max_s with a linear fade-out over the last fade_s (PCM16)."""
    sr, ch, sw, pcm = _wav_pcm(wb)
    if sw != 2:
        raise ValueError("only PCM16")
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    n = min(len(a), int(max_s * sr) * ch)
    a = a[:n]
    f = min(n, int(fade_s * sr) * ch)
    if f > 0:
        a[n - f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
    return _wav_bytes(sr, ch, sw, np.clip(a, -32768, 32767).astype(np.int16).tobytes())
