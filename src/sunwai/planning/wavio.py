"""Reading the 16 kHz mono PCM16 working wav without materialising it as float32."""

from __future__ import annotations

import os
import wave
from typing import Any, Union

import numpy as np
import numpy.typing as npt

SAMPLE_RATE = 16000


class Pcm16:
    """Read-only float32 view over a PCM16 wav's samples, backed by a memory map.

    Slicing converts only the slice, so the planner never holds the whole decoded track as float32. Supports
    ``len()``, ``.size``, integer/slice indexing and :meth:`to_array`.
    """

    def __init__(self, mm: npt.NDArray[np.int16] | np.memmap[Any, np.dtype[np.int16]]) -> None:
        """Wrap an int16 array or memmap."""
        self._mm = mm
        self.size = int(len(mm))
        self.shape = (self.size,)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, slice):
            return np.asarray(self._mm[key], dtype=np.float32) / 32768.0
        return float(self._mm[key]) / 32768.0

    def to_array(self) -> npt.NDArray[np.float32]:
        """The whole track as a float32 array."""
        return np.asarray(self[0 : self.size], dtype=np.float32)


Samples = Union[Pcm16, npt.NDArray[np.float32]]


def as_array(samples: Samples) -> npt.NDArray[np.float32]:
    """Materialise a :class:`Pcm16` view (or any array-like) as a float32 array."""
    if isinstance(samples, Pcm16):
        return samples.to_array()
    return np.asarray(samples, dtype=np.float32)


def wav_data_span(path: str) -> tuple[int, int]:
    """Return ``(offset, nbytes)`` of the RIFF ``data`` chunk, clamped to the real file size.

    Raises:
        RuntimeError: When the file is not 16 kHz mono PCM16 or has no data chunk.
    """
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != SAMPLE_RATE:
            raise RuntimeError(f"source16k.wav is not 16kHz mono PCM16: {path}")
    fsize = os.path.getsize(path)
    with open(path, "rb") as f:
        hdr = f.read(12)
        if hdr[:4] != b"RIFF" or hdr[8:12] != b"WAVE":
            raise RuntimeError(f"not a RIFF/WAVE file: {path}")
        pos = 12
        while pos + 8 <= fsize:
            f.seek(pos)
            cid, csz = f.read(4), int.from_bytes(f.read(4), "little")
            if cid == b"data":
                off = pos + 8
                return off, max(0, min(csz, fsize - off))
            pos += 8 + csz + (csz & 1)
    raise RuntimeError(f"no data chunk in {path}")


def read_mono16k(path: str) -> Pcm16:
    """Open a mono 16 kHz PCM16 wav as a lazy float32 view (``np.memmap`` underneath)."""
    off, nbytes = wav_data_span(path)
    n = nbytes // 2
    if n <= 0:
        return Pcm16(np.zeros(0, dtype=np.int16))
    return Pcm16(np.memmap(path, dtype="<i2", mode="r", offset=off, shape=(n,)))


def wav_ok(path: str, expect_dur: float | None = None) -> bool:
    """True for a 16 kHz mono PCM16 wav with a valid header, samples, and about the expected length."""
    try:
        if os.path.getsize(path) < 44:
            return False
        with wave.open(path, "rb") as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != SAMPLE_RATE:
                return False
            header_frames = w.getnframes()
        _off, nbytes = wav_data_span(path)
        n = nbytes // 2
        if n <= 0 or n < header_frames:  # header promises more samples than the file holds = truncated
            return False
        if expect_dur and expect_dur > 0 and n / float(SAMPLE_RATE) < expect_dur - max(2.0, 0.02 * expect_dur):
            return False
        return True
    except Exception:
        return False
