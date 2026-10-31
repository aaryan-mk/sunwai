"""Face detection, landmarks and affine alignment on top of LatentSync's InsightFace-based image processor."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

MIN_FACE_WIDTH = 50
MIN_FACE_HEIGHT = 80
MAX_ASPECT = 1.5
MIN_ASPECT = 0.2
LEFT_EYE_POINTS = [43, 48, 49, 51, 50]
RIGHT_EYE_POINTS = slice(101, 106)
NOSE_POINTS = [74, 77, 83, 86]


def is_usable_face(bbox: Sequence[int], score: float, threshold: float = 0.5) -> bool:
    """Apply LatentSync's face filter: minimum size, plausible aspect ratio and detector confidence.

    Args:
        bbox: ``(x1, y1, x2, y2)``.
        score: Detector confidence.
        threshold: Minimum confidence.

    Returns:
        True when the detection is a usable face.
    """
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    if width < MIN_FACE_WIDTH or height < MIN_FACE_HEIGHT:
        return False
    return MIN_ASPECT <= width / height <= MAX_ASPECT and score >= threshold


def landmarks3_from_106(landmarks: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Reduce 106 face landmarks to the three points used for alignment: left eye, right eye and nose.

    Args:
        landmarks: ``[106, 2]`` landmark array.

    Returns:
        ``[3, 2]`` rounded points.
    """
    left = np.mean(landmarks[LEFT_EYE_POINTS], axis=0)
    right = np.mean(landmarks[RIGHT_EYE_POINTS], axis=0)
    nose = np.mean(landmarks[NOSE_POINTS], axis=0)
    return np.round([left, right, nose])


class FaceAnalyzer:
    """Detects faces and aligns them to the square the UNet expects.

    Args:
        image_processor: LatentSync ``ImageProcessor`` (provides the face detector and the restorer).
        resolution: Side length of the aligned face.
    """

    def __init__(self, image_processor: Any, resolution: int) -> None:
        self.image_processor = image_processor
        self.resolution = resolution

    def detect_faces(self, frame_rgb: npt.NDArray[np.uint8], threshold: float = 0.5) -> list[dict[str, Any]]:
        """Detect ALL usable faces of a frame (not only the biggest).

        Args:
            frame_rgb: RGB frame.
            threshold: Minimum detector confidence.

        Returns:
            Dicts with ``bbox`` (x1, y1, x2, y2) and ``lm`` (3-point landmarks).
        """
        faces = []
        for face in self.image_processor.face_detector.app.get(frame_rgb):
            bbox = face.bbox.astype(np.int_).tolist()
            if is_usable_face(bbox, face.det_score, threshold):
                landmarks = np.round(face.landmark_2d_106).astype(np.int_)
                faces.append({"bbox": bbox, "lm": landmarks3_from_106(landmarks)})
        return faces

    def landmarks3(self, frame_rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64] | None:
        """3-point landmarks of the face LatentSync would edit, or None when there is none."""
        bbox, landmarks = self.image_processor.face_detector(frame_rgb)
        if bbox is None:
            return None
        return landmarks3_from_106(landmarks)

    def affine_from_landmarks(
        self, frame_rgb: npt.NDArray[np.uint8], lm3: npt.NDArray[Any]
    ) -> tuple[Any, list[int], Any]:
        """Align a face to the model resolution.

        Args:
            frame_rgb: RGB frame.
            lm3: 3-point landmarks of the face.

        Returns:
            ``(face [3, R, R] uint8 tensor, box, affine matrix)``.
        """
        import cv2
        import torch

        face, matrix = self.image_processor.restorer.align_warp_face(frame_rgb.copy(), landmarks3=lm3, smooth=True)
        box = [0, 0, face.shape[1], face.shape[0]]
        face = cv2.resize(face, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4)
        return torch.from_numpy(face).permute(2, 0, 1), box, matrix
