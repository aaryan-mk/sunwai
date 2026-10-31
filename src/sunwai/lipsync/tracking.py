"""Face tracks: link per-frame detections into tracks and cut the 224 px face crops SyncNet scores.

The geometry (IoU, linking, crop windows, padded slicing) is pure numpy so it can be tested without models;
only the final resize imports cv2, lazily.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

CROP_SIZE = 224
CROP_CONTEXT = 0.4
CROP_PAD_VALUE = 110
MEDIAN_KERNEL = 13

Box = Sequence[float]
Detection = dict[str, Any]


@dataclass
class FaceTrack:
    """One face followed across frames.

    Attributes:
        frame: Consecutive frame offsets covered by the track.
        bbox: ``[n, 4]`` boxes (x1, y1, x2, y2), interpolated over detection gaps.
        lm: Frame offset to the 3-point landmarks detected on that frame (``None`` when unknown).
    """

    frame: npt.NDArray[np.int_]
    bbox: npt.NDArray[np.float64]
    lm: dict[int, Any] = field(default_factory=dict)


@dataclass
class _OpenTrack:
    """A track still being extended while detections are linked."""

    ks: list[int]
    boxes: list[list[float]]
    lm: dict[int, Any]

    @property
    def last_k(self) -> int:
        return self.ks[-1]

    def extend(self, k: int, detection: Detection) -> None:
        """Append a detection made on frame offset ``k``."""
        self.ks.append(k)
        self.boxes.append([float(v) for v in detection["bbox"]])
        self.lm[k] = detection.get("lm")

    @classmethod
    def start(cls, k: int, detection: Detection) -> _OpenTrack:
        """Open a new track from a detection made on frame offset ``k``."""
        track = cls([], [], {})
        track.extend(k, detection)
        return track


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def link_tracks(
    dets: Sequence[Sequence[Detection]], min_track: int = 8, iou_th: float = 0.5, max_gap: int = 25, min_face: int = 30
) -> list[FaceTrack]:
    """Link the engine's per-frame face detections into tracks.

    Args:
        dets: ``dets[k]`` lists the detections (``bbox`` and optional ``lm``) of frame offset ``k``.
        min_track: Tracks with at most this many detections are dropped.
        iou_th: Minimum IoU between a track's last box and a detection to extend it.
        max_gap: Frames a track may go unseen before it is closed.
        min_face: Tracks whose mean face size is at most this (pixels) are dropped.

    Returns:
        Tracks sorted by their first frame.
    """
    active: list[_OpenTrack] = []
    closed: list[_OpenTrack] = []
    for k, frame_dets in enumerate(dets):
        used: set[int] = set()
        for track in sorted(active, key=lambda t: -t.last_k):
            match = _best_match(frame_dets, track.boxes[-1], used, iou_th)
            if match is not None and k - track.last_k <= max_gap:
                used.add(match)
                track.extend(k, frame_dets[match])
        active.extend(_OpenTrack.start(k, d) for j, d in enumerate(frame_dets) if j not in used)
        closed.extend(t for t in active if k - t.last_k > max_gap)
        active = [t for t in active if k - t.last_k <= max_gap]
    tracks = [_finish_track(t) for t in closed + active if len(t.ks) > min_track]
    kept = [t for t in tracks if _mean_face_size(t.bbox) > min_face]
    return sorted(kept, key=lambda t: t.frame[0])


def _best_match(frame_dets: Sequence[Detection], last_box: Box, used: set[int], iou_th: float) -> int | None:
    """Index of the unused detection overlapping ``last_box`` best (and above ``iou_th``)."""
    best, best_j = iou_th, None
    for j, det in enumerate(frame_dets):
        if j in used:
            continue
        overlap = iou(det["bbox"], last_box)
        if overlap > best:
            best, best_j = overlap, j
    return best_j


def _finish_track(track: _OpenTrack) -> FaceTrack:
    """Interpolate a track's boxes over the frames it was not detected on."""
    ks = np.array(track.ks)
    boxes = np.array(track.boxes)
    frames = np.arange(ks[0], ks[-1] + 1)
    interpolated = np.stack([np.interp(frames, ks, boxes[:, j]) for j in range(4)], 1)
    return FaceTrack(frame=frames, bbox=interpolated, lm=track.lm)


def _mean_face_size(bbox: npt.NDArray[np.float64]) -> float:
    """Larger of the mean box width and mean box height."""
    return float(max(np.mean(bbox[:, 2] - bbox[:, 0]), np.mean(bbox[:, 3] - bbox[:, 1])))


def crop_windows(bbox: npt.NDArray[np.float64], context: float = CROP_CONTEXT) -> npt.NDArray[np.int_]:
    """Compute, per frame, the square-ish source window ``(y0, y1, x0, x1)`` around a (median-smoothed) face box.

    Coordinates may fall outside the frame; :func:`slice_padded` fills those regions.

    Args:
        bbox: ``[n, 4]`` face boxes (x1, y1, x2, y2).
        context: Extra context around the face as a fraction of its half-size.

    Returns:
        ``[n, 4]`` integer windows.
    """
    from scipy import signal

    half = signal.medfilt(np.maximum(bbox[:, 3] - bbox[:, 1], bbox[:, 2] - bbox[:, 0]) / 2, MEDIAN_KERNEL)
    cy = signal.medfilt((bbox[:, 1] + bbox[:, 3]) / 2, MEDIAN_KERNEL)
    cx = signal.medfilt((bbox[:, 0] + bbox[:, 2]) / 2, MEDIAN_KERNEL)
    margin = (half * (1 + 2 * context)).astype(int)
    my, mx = cy + margin, cx + margin
    y0 = (my - half).astype(int) - margin
    y1 = (my + half * (1 + 2 * context)).astype(int) - margin
    x0 = (mx - half * (1 + context)).astype(int) - margin
    x1 = (mx + half * (1 + context)).astype(int) - margin
    return np.stack([y0, y1, x0, x1], 1)


def slice_padded(frame: npt.NDArray[np.uint8], window: Sequence[int]) -> npt.NDArray[np.uint8]:
    """Cut ``frame[y0:y1, x0:x1]`` as BGR, padding with grey where the window leaves the frame.

    Args:
        frame: RGB frame ``[H, W, 3]``.
        window: ``(y0, y1, x0, x1)`` source window.

    Returns:
        The BGR crop (the source is sliced first and only the slice is padded).
    """
    y0, y1, x0, x1 = (int(v) for v in window)
    height, width = frame.shape[:2]
    cy0, cy1, cx0, cx1 = max(0, y0), min(height, y1), max(0, x0), min(width, x1)
    face = frame[cy0:cy1, cx0:cx1, ::-1]
    if (cy0, cy1, cx0, cx1) != (y0, y1, x0, x1):
        pad = ((cy0 - y0, y1 - cy1), (cx0 - x0, x1 - cx1), (0, 0))
        face = np.pad(face, pad, "constant", constant_values=(CROP_PAD_VALUE, CROP_PAD_VALUE))
    return face


def face_crops(
    track: FaceTrack, frames: npt.NDArray[np.uint8], first_frame: int, context: float = CROP_CONTEXT
) -> npt.NDArray[np.uint8]:
    """Cut the 224 x 224 BGR face crops of a track.

    Args:
        track: The face track (frame offsets relative to ``first_frame``).
        frames: All chunk frames, RGB ``[F, H, W, 3]``.
        first_frame: Chunk frame index that offset 0 of the track refers to.
        context: Extra context around the face as a fraction of its half-size.

    Returns:
        ``[n, 224, 224, 3]`` uint8 crops.
    """
    import cv2

    windows = crop_windows(track.bbox, context)
    crops = [
        cv2.resize(
            np.ascontiguousarray(slice_padded(frames[first_frame + int(offset)], window)), (CROP_SIZE, CROP_SIZE)
        )
        for offset, window in zip(track.frame, windows, strict=True)
    ]
    return np.stack(crops)
