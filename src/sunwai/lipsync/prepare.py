"""Preparation of one run for denoising: aligned faces, boxes, affine matrices and whisper audio features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from sunwai.lipsync.faces import FaceAnalyzer
from sunwai.lipsync.media import cut_audio_run
from sunwai.lipsync.models import LatentSyncModels
from sunwai.lipsync.runs import FaceKey

MAX_MISSING_AUDIO_CHUNKS = 3


@dataclass
class Pack:
    """One face run travelling through the denoise and restore stages.

    Attributes:
        key: Identity of the face track the run belongs to.
        f0: First frame (inclusive).
        f1: End frame (exclusive).
        ids: Frame indices of the run.
        faces: Aligned faces ``[n, 3, R, R]`` (dropped once denoised).
        boxes: Face rectangles in aligned-face space.
        mats: Affine matrices.
        wch: Whisper feature chunk per frame (dropped once denoised).
        gen: Generated faces, set by the denoise stage.
        coff: Seam colour offsets, set by the denoise stage when paste-back v2 is on.
        n: Number of frames generated.
        offloaded: True when ``gen`` was moved to CPU to bound GPU memory.
    """

    key: FaceKey
    f0: int
    f1: int
    ids: list[int]
    faces: Any
    boxes: list[list[int]]
    mats: list[Any]
    wch: list[Any]
    gen: list[Any] | None = None
    coff: list[Any] | None = None
    n: int = 0
    offloaded: bool = field(default=False)


def fill_missing_landmarks(ids: Sequence[int], landmarks: Mapping[int, Any]) -> list[Any] | None:
    """Give every frame of a run landmarks, borrowing the nearest frame's where detection failed.

    Args:
        ids: Frame indices of the run.
        landmarks: Frame to 3-point landmarks (``None`` or absent where detection failed).

    Returns:
        One landmark set per frame, or None when no frame of the run has landmarks.
    """
    valid = [i for i in ids if landmarks.get(i) is not None]
    if not valid:
        return None
    return [landmarks[min(valid, key=lambda j: abs(j - i))] if landmarks.get(i) is None else landmarks[i] for i in ids]


def pad_audio_chunks(chunks: list[Any], n_frames: int) -> list[Any]:
    """Repeat the last audio chunk when whisper produced up to three fewer chunks than frames.

    Args:
        chunks: Whisper feature chunks.
        n_frames: Frames in the run.

    Returns:
        At least ``n_frames`` chunks.

    Raises:
        RuntimeError: More than three chunks are missing.
    """
    missing = n_frames - len(chunks)
    if missing <= 0:
        return chunks
    if missing > MAX_MISSING_AUDIO_CHUNKS:
        raise RuntimeError(f"whisper chunks {len(chunks)} << frames {n_frames}")
    return chunks + [chunks[-1]] * missing


class RunPreparer:
    """Aligns faces and extracts audio features for runs.

    Args:
        faces: Face analyzer used for alignment.
        models: Resident models (whisper encoder, image processor).
        fps: Video frame rate.
    """

    def __init__(self, faces: FaceAnalyzer, models: LatentSyncModels, fps: int) -> None:
        self.faces = faces
        self.models = models
        self.fps = fps

    def prepare(
        self,
        frames: npt.NDArray[np.uint8],
        key: FaceKey,
        start: int,
        end: int,
        landmarks: Mapping[int, Any],
        dub_wav: str,
        wav_path: str,
        audio_offset_ms: float,
    ) -> Pack | None:
        """Prepare the run ``[start, end)``.

        Args:
            frames: All chunk frames.
            key: Face track identity.
            start: First frame (inclusive).
            end: End frame (exclusive).
            landmarks: Frame to 3-point landmarks.
            dub_wav: Dub audio of the chunk.
            wav_path: Scratch wav this run's audio is cut to.
            audio_offset_ms: Audio offset in milliseconds.

        Returns:
            The prepared pack, or None when no frame of the run has landmarks.
        """
        import torch

        ids = list(range(start, end))
        run_landmarks = fill_missing_landmarks(ids, landmarks)
        if run_landmarks is None:
            return None
        self.models.image_processor.restorer.p_bias = None
        aligned = [self.faces.affine_from_landmarks(frames[i], lm) for i, lm in zip(ids, run_landmarks, strict=True)]
        cut_audio_run(dub_wav, start, end - start, self.fps, audio_offset_ms, wav_path)
        audio = self.models.audio_encoder
        chunks = audio.feature2chunks(feature_array=audio.audio2feat(wav_path), fps=self.fps)
        return Pack(
            key=key,
            f0=start,
            f1=end,
            ids=ids,
            faces=torch.stack([face for face, _, _ in aligned]),
            boxes=[box for _, box, _ in aligned],
            mats=[matrix for _, _, matrix in aligned],
            wch=pad_audio_chunks(chunks, len(ids)),
        )
