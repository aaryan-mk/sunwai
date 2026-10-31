"""Acoustic speaker features (ECAPA centroid + median F0) used by the planner and the bank builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sunwai.planning.voices.catalogue import ECAPA_ID
from sunwai.planning.voices.types import Features

SAMPLE_RATE = 16000
Interval = tuple[float, float]
_ECAPA: Any = None


def ecapa(savedir: str | Path) -> Any:
    """Return the cached ECAPA speaker encoder, loading it on first use (CPU)."""
    global _ECAPA
    if _ECAPA is None:
        from speechbrain.inference.speaker import EncoderClassifier

        _ECAPA = EncoderClassifier.from_hparams(source=ECAPA_ID, savedir=str(savedir), run_opts={"device": "cpu"})
    return _ECAPA


def embed(y16: Any, savedir: str | Path) -> Any:
    """Return the unit-norm ECAPA embedding of 16 kHz mono audio."""
    import numpy as np
    import torch

    with torch.no_grad():
        waveform = torch.from_numpy(np.ascontiguousarray(y16, dtype="float32"))[None]
        vector = ecapa(savedir).encode_batch(waveform).squeeze().cpu().numpy()
    return vector / (np.linalg.norm(vector) + 1e-9)


def median_logf0(y16: Any) -> float | None:
    """Return the median ln F0 of voiced frames, or None when fewer than 20 voiced frames exist."""
    import librosa
    import numpy as np

    f0, voiced, _ = librosa.pyin(np.asarray(y16, dtype="float32"), fmin=60, fmax=450, sr=SAMPLE_RATE, frame_length=1024)
    f0 = f0[voiced & ~np.isnan(f0)]
    return float(np.median(np.log(f0))) if len(f0) >= 20 else None


def subtract_intervals(span: Interval, others: list[Interval]) -> list[Interval]:
    """Remove every interval of ``others`` from ``span``, returning the remaining pieces in order."""
    pieces = [span]
    for start, end in others:
        remaining: list[Interval] = []
        for x, z in pieces:
            if end <= x or start >= z:
                remaining.append((x, z))
                continue
            if start > x:
                remaining.append((x, start))
            if end < z:
                remaining.append((end, z))
        pieces = remaining
    return pieces


def speech_pieces(turns: list[dict[str, Any]], speaker: Any, min_len: float = 1.0) -> tuple[list[Interval], bool]:
    """Pick the speaker's evidence pieces: clean (non-overlapping) speech, else any turn of at least ``min_len``.

    Returns:
        ``(pieces sorted longest first, clean)`` where ``clean`` says the pieces are overlap-free.
    """
    others = sorted((t["start"], t["end"]) for t in turns if t["speaker"] != speaker)
    pieces: list[Interval] = []
    for t in turns:
        if t["speaker"] == speaker:
            pieces += [p for p in subtract_intervals((t["start"], t["end"]), others) if p[1] - p[0] >= min_len]
    clean = bool(pieces)
    if not pieces:
        pieces = [(t["start"], t["end"]) for t in turns if t["speaker"] == speaker and t["end"] - t["start"] >= min_len]
    pieces.sort(key=lambda p: (-(p[1] - p[0]), p[0]))
    return pieces, clean


def speaker_features(
    y: Any, turns: list[dict[str, Any]], savedir: str | Path, max_total: float = 40.0, max_piece: float = 8.0
) -> dict[Any, Features | None]:
    """Compute per speaker an ECAPA centroid and median ln F0 from clean speech (any >= 1 s turn as fallback).

    Args:
        y: 16 kHz mono waveform.
        turns: ``[{speaker, start, end}]`` diarization turns (seconds).
        savedir: Where the ECAPA model is cached.
        max_total: Seconds of evidence collected per speaker.
        max_piece: Longest single piece (s) embedded.

    Returns:
        ``{speaker: {emb, logf0, evidence_s, clean}}``; None for a speaker without usable speech.
    """
    import numpy as np

    out: dict[Any, Features | None] = {}
    for speaker in sorted({t["speaker"] for t in turns}):
        pieces, clean = speech_pieces(turns, speaker)
        embeddings, weights, total, audio = [], [], 0.0, []
        for x, z in pieces:
            if total >= max_total:
                break
            z = min(z, x + max_piece)
            segment = y[int(x * SAMPLE_RATE) : int(z * SAMPLE_RATE)]
            embeddings.append(embed(segment, savedir))
            weights.append(z - x)
            total += z - x
            audio.append(segment)
        if not embeddings:
            out[speaker] = None
            continue
        centroid = np.average(np.stack(embeddings), axis=0, weights=weights)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        logf0 = median_logf0(np.concatenate(audio)[: SAMPLE_RATE * 20])
        out[speaker] = {"emb": centroid, "logf0": logf0, "evidence_s": round(total, 2), "clean": clean}
    return out
