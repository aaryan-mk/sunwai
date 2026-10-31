"""Active-speaker gate for the lip-sync engine.

Decides, per frame, whether the face that LatentSync would re-animate is actually speaking in the ORIGINAL video.
Evidence is the SyncNet v2 audio-visual confidence of the face track against the original audio, computed on
sliding 2 s windows (:mod:`sunwai.lipsync.syncnet`). Static faces (stills, frozen frames) are decided by mouth
motion instead. Kill switch: env ``LIP_ASD=0``. The engine fails open on any exception.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from sunwai.lipsync.flags import env_enabled, env_opt_in
from sunwai.lipsync.syncnet import Scores, SyncNetScorer
from sunwai.lipsync.tracking import Detection, FaceTrack, face_crops, link_tracks

MOTION_SMOOTHING = 25
MOUTH_ROWS = slice(120, 210)
MOUTH_COLS = slice(50, 174)
S3FD_CONF = 0.9
S3FD_SCALES = [0.25]


def gate_enabled() -> bool:
    """Return False when the gate is switched off with env ``LIP_ASD``."""
    return env_enabled("LIP_ASD")


@dataclass(frozen=True)
class GateConfig:
    """Thresholds of the speaking decision.

    Attributes:
        conf_th: Minimum SyncNet confidence for a speaking face.
        off_max: Largest absolute audio-visual offset (frames) still counted as in sync.
        static_th: Mouth-motion threshold below which a face counts as static.
        static_sync: Also lip-sync static faces (default off: a still photo gets no mouth movement).
    """

    conf_th: float = 2.7
    off_max: int = 8
    static_th: float = 0.05
    static_sync: bool = False

    @classmethod
    def from_env(cls, conf_th: float | None = None) -> GateConfig:
        """Read thresholds from ``LIP_ASD_CONF``, ``LIP_ASD_OFFMAX``, ``LIP_ASD_STATIC`` and ``LIP_ASD_STATIC_SYNC``.

        Args:
            conf_th: Explicit confidence threshold overriding the environment.

        Returns:
            The configuration.
        """
        return cls(
            conf_th=float(conf_th if conf_th is not None else os.environ.get("LIP_ASD_CONF", "2.7")),
            off_max=int(os.environ.get("LIP_ASD_OFFMAX", "8")),
            static_th=float(os.environ.get("LIP_ASD_STATIC", "0.05")),
            static_sync=env_opt_in("LIP_ASD_STATIC_SYNC"),
        )


class TrackScorer(Protocol):
    """Anything that scores face crops against audio (SyncNet in production, a fake in tests)."""

    def score_track(self, crops: npt.NDArray[np.uint8], wav: npt.NDArray[np.int16], t0_frame: int) -> Scores:
        """Return per-frame ``(confidence, offset)`` for the crops."""
        ...


CropFn = Callable[[FaceTrack, npt.NDArray[np.uint8], int], npt.NDArray[np.uint8]]


@dataclass
class SpeakingTrack:
    """A face track with its speaking decision.

    Attributes:
        tid: Track index within the window.
        frame: Absolute chunk frame indices of the track.
        lm: Absolute frame index to 3-point landmarks.
        mask: True where the face is speaking in the original video.
        conf: SyncNet confidence per frame.
        off: SyncNet audio-visual offset per frame.
        mot: Mouth motion per frame.
        bbox: Face boxes per frame.
    """

    tid: int
    frame: npt.NDArray[np.int_]
    lm: dict[int, Any]
    mask: npt.NDArray[np.bool_]
    conf: npt.NDArray[np.float64]
    off: npt.NDArray[np.float64]
    mot: npt.NDArray[np.float32]
    bbox: npt.NDArray[np.float64]


def mouth_motion(crops: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """Per-frame inter-frame motion of the mouth region (lower-centre of the 224 crop), median-smoothed.

    A still photo or frozen frame scores about 0 even though SyncNet cannot score it.

    Args:
        crops: ``[T, 224, 224, 3]`` BGR crops.

    Returns:
        ``[T]`` mean absolute grey difference to the previous frame.
    """
    import cv2
    from scipy import signal

    gray = np.stack(
        [cv2.cvtColor(np.ascontiguousarray(c[MOUTH_ROWS, MOUTH_COLS]), cv2.COLOR_BGR2GRAY) for c in crops]
    ).astype(np.float32)
    diff = np.zeros(len(gray), np.float32)
    if len(gray) > 1:
        diff[1:] = np.abs(gray[1:] - gray[:-1]).mean((1, 2))
        diff[0] = diff[1]
    kernel = min(MOTION_SMOOTHING, len(diff)) | 1
    return np.asarray(signal.medfilt(diff, kernel), np.float32) if len(diff) >= 3 else diff


def speaking_mask(
    conf: npt.NDArray[np.float64], off: npt.NDArray[np.float64], motion: npt.NDArray[np.float32], config: GateConfig
) -> npt.NDArray[np.bool_]:
    """Combine SyncNet evidence and mouth motion into a per-frame speaking decision.

    Args:
        conf: SyncNet confidence per frame (NaN where undefined).
        off: SyncNet offset per frame.
        motion: Mouth motion per frame.
        config: Thresholds.

    Returns:
        True where the face is speaking; static faces follow ``config.static_sync``.
    """
    synced = (~np.isnan(conf)) & (conf >= config.conf_th) & (np.abs(np.nan_to_num(off, nan=99)) <= config.off_max)
    return np.asarray(np.where(motion < config.static_th, config.static_sync, synced), bool)


@dataclass
class ScoredTrack:
    """A tracked face with all per-frame evidence."""

    track: FaceTrack
    conf: npt.NDArray[np.float64]
    off: npt.NDArray[np.float64]
    mot: npt.NDArray[np.float32]


def nearest_track_sample(
    scored: Sequence[ScoredTrack], frame_offset: int, center: Sequence[float]
) -> tuple[float, float, float] | None:
    """Pick the evidence of the track whose box contains ``center`` (the edited face's nose point).

    Args:
        scored: Scored tracks (frame offsets relative to the window start).
        frame_offset: Frame offset within the window.
        center: ``(x, y)`` of the face LatentSync would edit.

    Returns:
        ``(conf, off, motion)`` of the closest containing track, or None when no track contains the point.
    """
    best: tuple[float, tuple[float, float, float]] | None = None
    for item in scored:
        k = frame_offset - int(item.track.frame[0])
        if k < 0 or k >= len(item.track.frame):
            continue
        x1, y1, x2, y2 = item.track.bbox[k]
        if x1 <= center[0] <= x2 and y1 <= center[1] <= y2:
            dist = (center[0] - (x1 + x2) / 2) ** 2 + (center[1] - (y1 + y2) / 2) ** 2
            if best is None or dist < best[0]:
                best = (dist, (item.conf[k], item.off[k], item.mot[k]))
    return None if best is None else best[1]


class S3fdTracker:
    """Legacy S3FD face tracker (second detection pass), built lazily on first use.

    Args:
        device: Torch device for the detector.
        detector: Existing ``SyncNetDetector`` to reuse instead of building one.
    """

    def __init__(self, device: str = "cuda", detector: Any | None = None) -> None:
        self._device = device
        self._detector = detector

    @property
    def detector(self) -> Any:
        """The ``SyncNetDetector`` (built on first access)."""
        if self._detector is None:
            from eval.syncnet_detect import SyncNetDetector  # type: ignore[import-not-found]

            self._detector = SyncNetDetector(
                device=self._device, detect_results_dir=os.path.join(os.environ.get("TMPDIR", "/tmp"), "asd_det")
            )
        return self._detector

    def tracks(self, frames: npt.NDArray[np.uint8], first: int, last: int) -> list[dict[str, Any]]:
        """Detect and track S3FD faces over frames ``[first, last)``."""
        dets = []
        for f in range(first, last):
            boxes = self.detector.s3f_detector.detect_faces(frames[f], conf_th=S3FD_CONF, scales=S3FD_SCALES)
            dets.append([{"frame": f - first, "bbox": b[:-1].tolist(), "conf": b[-1]} for b in boxes])
        tracks: list[dict[str, Any]] = self.detector.track_face(dets, min_track=8, min_face_size=30)
        return tracks


class ActiveSpeakerGate:
    """Decides which tracked faces are speaking in the original video.

    Args:
        scorer: SyncNet (or any :class:`TrackScorer`) used to score face tracks.
        config: Decision thresholds.
        tracker: Legacy S3FD tracker for :meth:`speaking`; built lazily when omitted.
        crop_fn: Cuts the 224 px crops of a track (replaceable in tests).
    """

    def __init__(
        self,
        scorer: TrackScorer,
        config: GateConfig,
        tracker: S3fdTracker | None = None,
        crop_fn: CropFn = face_crops,
    ) -> None:
        self.scorer = scorer
        self.config = config
        self.tracker = tracker or S3fdTracker()
        self._crop = crop_fn

    @classmethod
    def load(cls, device: str = "cuda", detector: Any | None = None, conf_th: float | None = None) -> ActiveSpeakerGate:
        """Build a gate backed by the SyncNet checkpoint on disk and thresholds from the environment."""
        return cls(
            SyncNetScorer(device=device),
            GateConfig.from_env(conf_th),
            S3fdTracker(device, detector),
        )

    @property
    def conf_th(self) -> float:
        """Confidence threshold in use."""
        return self.config.conf_th

    def speaking_tracks(
        self,
        frames: npt.NDArray[np.uint8],
        wav: npt.NDArray[np.int16],
        first: int,
        last: int,
        dets: Sequence[Sequence[Detection]],
    ) -> list[SpeakingTrack]:
        """Multi-face gate: score every face track independently against the ORIGINAL audio.

        Args:
            frames: All chunk frames.
            wav: Original audio of the chunk.
            first: First absolute frame of the window.
            last: End of the window (exclusive).
            dets: Engine detections for frames ``first .. last - 1``.

        Returns:
            One decision per linked track, in absolute frame numbers.
        """
        out = []
        for tid, track in enumerate(link_tracks(dets)):
            crops = self._crop(track, frames, first)
            conf, off = self.scorer.score_track(crops, wav, first + int(track.frame[0]))
            motion = mouth_motion(crops)
            out.append(
                SpeakingTrack(
                    tid=tid,
                    frame=track.frame + first,
                    lm={first + k: v for k, v in track.lm.items()},
                    mask=speaking_mask(conf, off, motion, self.config),
                    conf=conf,
                    off=off,
                    mot=motion,
                    bbox=track.bbox,
                )
            )
        return out

    def speaking(
        self,
        frames: npt.NDArray[np.uint8],
        wav: npt.NDArray[np.int16],
        first: int,
        last: int,
        centers: Mapping[int, Sequence[float]],
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
        """Single-face gate: decide per frame whether the face LatentSync would edit is speaking.

        Args:
            frames: All chunk frames.
            wav: Original audio of the chunk.
            first: First absolute frame of the window.
            last: End of the window (exclusive).
            centers: Frame to ``(x, y)`` of the edited face's nose point.

        Returns:
            ``(mask, confs)`` over ``[first, last)``; mask True means keep the lip-sync.
        """
        scored = []
        for track_dict in self.tracker.tracks(frames, first, last):
            track = FaceTrack(frame=track_dict["frame"], bbox=track_dict["bbox"])
            crops = self._crop(track, frames, first)
            track_conf, track_off = self.scorer.score_track(crops, wav, first + int(track.frame[0]))
            scored.append(ScoredTrack(track, track_conf, track_off, mouth_motion(crops)))
        mask = np.zeros(last - first, bool)
        confs = np.full(last - first, np.nan)
        for f in range(first, last):
            center = centers.get(f)
            sample = None if center is None else nearest_track_sample(scored, f - first, center)
            if sample is None:
                continue
            conf, off, motion = sample
            confs[f - first] = conf
            if motion < self.config.static_th:
                mask[f - first] = self.config.static_sync
            elif not np.isnan(conf):
                mask[f - first] = conf >= self.config.conf_th and abs(off) <= self.config.off_max
        return mask, confs
