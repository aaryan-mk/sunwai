"""SyncNet v2 audio-visual scoring of face tracks against the ORIGINAL audio.

The score windows and their aggregation are pure numpy (:func:`window_starts`, :func:`average_window_scores`);
torch and the SyncNet checkpoint are only touched inside :class:`SyncNetScorer`, lazily.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

FPS = 25
SAMPLE_RATE = 16000
LIP_WINDOW = 5
VSHIFT = 15
MIN_TRACK_FRAMES = 8
MIN_AUDIO_SAMPLES = 3200
SAMPLES_PER_FRAME = 640
DEFAULT_SYNCNET_PATH = "checkpoints/auxiliary/syncnet_v2.model"

Scores = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]


def window_starts(n: int, window: int, stride: int) -> list[int]:
    """Start offsets of the sliding score windows over ``n`` positions, always covering the last position.

    Args:
        n: Number of scoreable positions.
        window: Window length.
        stride: Distance between window starts.

    Returns:
        Ascending window starts.
    """
    starts = list(range(0, max(1, n - window + 1), stride))
    if starts[-1] + window < n:
        starts.append(n - window)
    return starts


def average_window_scores(window_scores: Sequence[tuple[int, float, int]], length: int, window: int) -> Scores:
    """Spread per-window ``(confidence, offset)`` scores over frames and average where windows overlap.

    Each window starting at ``a`` covers ``window + 4`` frames (the last positions have no full 5-frame lip
    window). Frames no window reaches stay NaN, except a trailing undefined tail which inherits the last decision so
    the gate does not see a spurious boundary.

    Args:
        window_scores: ``(start, confidence, offset)`` per window.
        length: Number of frames in the track.
        window: Window length used to produce the scores.

    Returns:
        Per-frame ``(confidence, offset)`` arrays of ``length`` (NaN where undefined).
    """
    conf = np.full(length, np.nan)
    off = np.full(length, np.nan)
    count = np.zeros(length)
    conf_sum = np.zeros(length)
    off_sum = np.zeros(length)
    for start, c, o in window_scores:
        span = slice(start, start + window + 4)
        conf_sum[span] += c
        off_sum[span] += o
        count[span] += 1
    covered = count > 0
    conf[covered] = conf_sum[covered] / count[covered]
    off[covered] = off_sum[covered] / count[covered]
    last = np.where(covered)[0]
    if len(last) and last[-1] < length - 1:
        conf[last[-1] + 1 :] = conf[last[-1]]
        off[last[-1] + 1 :] = off[last[-1]]
    return conf, off


def _undefined(length: int) -> Scores:
    """Return all-NaN score arrays."""
    nan = np.full(length, np.nan)
    return nan, nan.copy()


class SyncNetScorer:
    """Scores face-crop tracks with SyncNet over sliding windows.

    Args:
        device: Torch device string for the model.
        syncnet_path: SyncNet v2 checkpoint.
        window: Sliding window length in frames (2 s at 25 fps).
        stride: Distance between window starts.
        batch_size: Lip windows per forward pass; defaults to env ``LIP_ASD_BS`` (64).
    """

    def __init__(
        self,
        device: str = "cuda",
        syncnet_path: str = DEFAULT_SYNCNET_PATH,
        window: int = 50,
        stride: int = 5,
        batch_size: int | None = None,
    ) -> None:
        from eval.syncnet import SyncNetEval  # type: ignore[import-not-found]

        self.device = device
        self._net = SyncNetEval(device=device)
        self._net.loadParameters(syncnet_path)
        self._net.__S__.eval()
        self.window = window
        self.stride = stride
        self.batch_size = int(batch_size if batch_size is not None else os.environ.get("LIP_ASD_BS", "64"))
        self._mfcc_cache: tuple[tuple[int, int, int], npt.NDArray[np.float32], Any] | None = None

    def _mfcc(self, wav: npt.NDArray[np.int16]) -> npt.NDArray[np.float32]:
        """MFCCs of the whole chunk audio, computed once and sliced per track (4 MFCC frames per video frame)."""
        import python_speech_features

        key = (id(wav), len(wav), int(np.asarray(wav[::1600], np.int64).sum()))
        cached = self._mfcc_cache
        # Strong reference to `wav`: a recycled id() can never hit a stale cache entry.
        if cached is None or cached[0] != key or cached[2] is not wav:
            features = np.ascontiguousarray(python_speech_features.mfcc(wav, SAMPLE_RATE).T.astype(np.float32))
            self._mfcc_cache = (key, features, wav)
        return self._mfcc_cache[1]  # type: ignore[index]

    def score_track(self, crops: npt.NDArray[np.uint8], wav: npt.NDArray[np.int16], t0_frame: int) -> Scores:
        """Score one track against the original audio.

        Args:
            crops: ``[T, 224, 224, 3]`` BGR face crops.
            wav: int16 16 kHz mono audio of the whole chunk.
            t0_frame: Chunk frame index of ``crops[0]``.

        Returns:
            Per-frame ``(confidence, offset)`` arrays of length ``T`` (NaN where undefined).
        """
        import torch

        length = len(crops)
        a0 = int(round(t0_frame / FPS * SAMPLE_RATE))
        audio = wav[a0 : a0 + int(round(length / FPS * SAMPLE_RATE))]
        if length < MIN_TRACK_FRAMES or len(audio) < MIN_AUDIO_SAMPLES:
            return _undefined(length)
        n = min(length, int(len(audio) // SAMPLES_PER_FRAME)) - LIP_WINDOW
        if n < 3:
            return _undefined(length)
        mfcc = self._mfcc(wav)[:, 4 * t0_frame : 4 * t0_frame + 4 * length]
        with torch.no_grad():
            distances = self._distances(crops, torch.from_numpy(mfcc[None, None]), n)
            w = min(self.window, n)
            window_scores = []
            for start in window_starts(n, w, self.stride):
                mean_dist = distances[start : start + w].mean(0)
                window_scores.append(
                    (start, (mean_dist.median() - mean_dist.min()).item(), VSHIFT - int(mean_dist.argmin()))
                )
        return average_window_scores(window_scores, length, w)

    def _distances(self, crops: npt.NDArray[np.uint8], mfcc: Any, n: int) -> Any:
        """Lip/audio embedding distances ``[n, 2 * VSHIFT + 1]`` for the first ``n`` lip windows."""
        import torch

        images = torch.from_numpy(np.ascontiguousarray(crops)).to(self.device)
        net = self._net.__S__
        lip_embeddings, audio_embeddings = [], []
        for i in range(0, n, self.batch_size):
            idx = range(i, min(n, i + self.batch_size))
            lips = torch.stack([images[v : v + LIP_WINDOW] for v in idx]).permute(0, 4, 1, 2, 3).float()
            audio = torch.cat([mfcc[:, :, :, v * 4 : v * 4 + 20] for v in idx], 0).to(self.device)
            lip_embeddings.append(net.forward_lip(lips).cpu())
            audio_embeddings.append(net.forward_aud(audio).cpu())
        return _shifted_distances(torch.cat(lip_embeddings), torch.cat(audio_embeddings))


def _shifted_distances(lip: Any, audio: Any, vshift: int = VSHIFT) -> Any:
    """Distance between each lip embedding and the audio embeddings shifted by up to ``vshift`` frames."""
    import torch

    shifted = torch.nn.functional.pad(audio, (0, 0, vshift, vshift)).unfold(0, 2 * vshift + 1, 1)[: lip.shape[0]]
    return (lip[:, :, None] - shifted).norm(dim=1)
