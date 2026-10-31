"""ROI paste-back of the generated mouth region onto the original frames, with seam colour matching.

Paste-back v2 composites only the generated lower-face region (feathered, colour-offset at the seam) onto the
ORIGINAL frame. The older path replaced the whole aligned face rectangle with a resampled 512 px copy, which
softened the face and left a halo outside the mouth.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

ROI_MARGIN_LOW = 3
ROI_MARGIN_HIGH = 4
SEAM_ERODE_KERNEL = 33
FEATHER_ERODE_KERNEL = 9
FEATHER_BLUR = (25, 25)
FEATHER_SIGMA = (7.0, 7.0)
MAX_COLOR_OFFSET = 0.06
AGREE_THRESHOLD = 0.25
MIN_SEAM_PIXELS = 300


def roi_bounds(corners: npt.NDArray[np.float64], width: int, height: int) -> tuple[int, int, int, int] | None:
    """Bounding box of the pasted rectangle in frame space, widened by a small margin and clipped to the frame.

    Args:
        corners: ``[4, 2]`` (x, y) corners of the pasted rectangle in frame coordinates.
        width: Frame width.
        height: Frame height.

    Returns:
        ``(x0, y0, x1, y1)``, or None when the rectangle lies outside the frame.
    """
    x0 = max(0, int(np.floor(corners[:, 0].min())) - ROI_MARGIN_LOW)
    y0 = max(0, int(np.floor(corners[:, 1].min())) - ROI_MARGIN_LOW)
    x1 = min(width, int(np.ceil(corners[:, 0].max())) + ROI_MARGIN_HIGH)
    y1 = min(height, int(np.ceil(corners[:, 1].max())) + ROI_MARGIN_HIGH)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def crossfade(restored: npt.NDArray[Any], base: npt.NDArray[Any], weight: float) -> npt.NDArray[np.uint8]:
    """Blend a generated frame over the base frame; ``weight`` is the share of the generated pixels."""
    mixed = weight * np.asarray(restored, np.float32) + (1 - weight) * base
    return np.asarray(np.clip(mixed, 0, 255).astype(np.uint8))


class PasteBack:
    """Restores generated faces into frames.

    Args:
        device: Torch device the generated faces live on.
        image_processor: LatentSync ``ImageProcessor`` (provides ``mask_image`` and the legacy ``restorer``).
        paste_v2: Use the ROI paste-back; False falls back to the whole-rectangle restorer.
    """

    def __init__(self, device: Any, image_processor: Any, paste_v2: bool = True) -> None:
        self.device = device
        self.image_processor = image_processor
        self.paste_v2_enabled = paste_v2
        self._constants: tuple[Any, Any] | None = None

    def _paste_constants(self) -> tuple[Any, Any]:
        """Seam band and feathered mask derived from the fixed mouth mask (computed once)."""
        if self._constants is None:
            import kornia
            import torch.nn.functional as tf

            # 1 = region the model generates.
            generated = (1 - self.image_processor.mask_image[0:1]).to(self.device).float()[None]
            binary = (generated > 0.5).float()
            eroded = -tf.max_pool2d(-binary, SEAM_ERODE_KERNEL, 1, SEAM_ERODE_KERNEL // 2)
            band = (binary - eroded).clamp(0, 1)[0]
            core = -tf.max_pool2d(-binary, FEATHER_ERODE_KERNEL, 1, FEATHER_ERODE_KERNEL // 2)
            feather = kornia.filters.gaussian_blur2d(core, FEATHER_BLUR, FEATHER_SIGMA)[0, 0]
            self._constants = (band, feather)
        return self._constants

    def color_offsets(self, gen_faces: Sequence[Any], faces_u8: Sequence[Any]) -> list[Any]:
        """Per-frame, per-channel mean colour offset (original minus generated) at the seam band, in RGB ``[-1, 1]``.

        Only skin pixels of the original count (YCrCb skin test) and only where generated and original already agree,
        so hair, ear, neck or background in the band cannot tint the mouth patch. Offsets are clamped to +-0.06 (about
        8 levels); fewer than 300 valid pixels gives no offset. The offset is only applied inside the feathered mouth
        mask (:meth:`paste_v2`).

        Args:
            gen_faces: Generated faces ``[3, R, R]`` in ``[-1, 1]``.
            faces_u8: Original aligned faces ``[3, R, R]`` uint8.

        Returns:
            One ``[3]`` offset tensor per face (stays on the device, no CUDA sync).
        """
        import torch

        band, _ = self._paste_constants()
        offsets = []
        for generated, original in zip(gen_faces, faces_u8, strict=True):
            with torch.no_grad():
                offsets.append(self._seam_offset(band, generated, original))
        return offsets

    def _seam_offset(self, band: Any, generated: Any, original: Any) -> Any:
        """Seam colour offset ``[3]`` of one face (see :meth:`color_offsets`)."""
        import torch

        rgb = original.to(self.device).float()
        luma = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        cr = (rgb[0] - luma) * 0.713 + 128
        cb = (rgb[2] - luma) * 0.564 + 128
        skin = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)).float()[None]
        diff = (rgb / 255 * 2 - 1) - generated.float()
        weight = band * skin * (diff.abs().max(0, keepdim=True)[0] < AGREE_THRESHOLD).float()
        count = weight.sum()
        mean = (diff * weight).sum((1, 2)) / count.clamp_min(1)
        return torch.where(
            count >= MIN_SEAM_PIXELS, mean.clamp(-MAX_COLOR_OFFSET, MAX_COLOR_OFFSET), torch.zeros_like(mean)
        )

    def paste_v2(
        self, face: Any, frame: npt.NDArray[np.uint8], box: Sequence[int], matrix: Any, offset: Any
    ) -> npt.NDArray[np.uint8]:
        """ROI paste-back: warp and composite only the destination bounding box of the face rectangle.

        This uses about 100x less GPU memory than a full-frame warp at 1080p/4K.

        Args:
            face: Generated face ``[3, R, R]`` in ``[-1, 1]``.
            frame: Original RGB frame.
            box: ``(x1, y1, x2, y2)`` of the face rectangle in aligned-face space.
            matrix: Affine matrix from frame to aligned-face coordinates.
            offset: ``[3]`` colour offset from :meth:`color_offsets`.

        Returns:
            A copy of ``frame`` with the face composited in.
        """
        import torch

        with torch.no_grad():
            return self._paste_v2(face, frame, box, matrix, offset)

    def _paste_v2(
        self, face: Any, frame: npt.NDArray[np.uint8], box: Sequence[int], matrix: Any, offset: Any
    ) -> npt.NDArray[np.uint8]:
        """Body of :meth:`paste_v2`, run without autograd."""
        import torch
        import torch.nn.functional as tf
        from kornia.geometry.transform import invert_affine_transform, warp_affine

        _, feather = self._paste_constants()
        x1, y1, x2, y2 = box
        h, w = int(y2 - y1), int(x2 - x1)
        height, width = frame.shape[:2]
        matrix_t = matrix if not isinstance(matrix, np.ndarray) else torch.from_numpy(matrix).to(self.device)[None]
        inverse = invert_affine_transform(matrix_t.to(torch.float32))  # aligned-face -> frame coordinates, [1,2,3]
        forward = inverse[0].cpu().numpy().astype(np.float64)
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float64) @ forward.T
        bounds = roi_bounds(corners, width, height)
        if bounds is None:
            return np.ascontiguousarray(frame).copy()
        rx0, ry0, rx1, ry1 = bounds
        generated = (face.float() + offset[:, None, None]).clamp(-1, 1)
        generated = tf.interpolate(generated[None], size=(h, w), mode="bicubic", align_corners=False)
        mask = tf.interpolate(feather[None, None], size=(h, w), mode="bilinear", align_corners=False)
        inverse_roi = inverse.clone()  # same transform, ROI-local destination coordinates
        inverse_roi[0, 0, 2] -= rx0
        inverse_roi[0, 1, 2] -= ry0
        roi_size = (ry1 - ry0, rx1 - rx0)
        warped = warp_affine(generated, inverse_roi, roi_size, padding_mode="zeros")[0]
        warped_mask = warp_affine(mask, inverse_roi, roi_size, padding_mode="zeros")[0]
        original = (
            torch.from_numpy(np.ascontiguousarray(frame[ry0:ry1, rx0:rx1])).to(self.device).float().permute(2, 0, 1)
        )
        blended = warped_mask * ((warped / 2 + 0.5).clamp(0, 1) * 255) + (1 - warped_mask) * original
        result = np.ascontiguousarray(frame).copy()
        result[ry0:ry1, rx0:rx1] = blended.permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        return result

    def restore(
        self,
        gen_faces: Sequence[Any],
        frames: Sequence[npt.NDArray[np.uint8]],
        boxes: Sequence[Sequence[int]],
        mats: Sequence[Any],
        coffs: Sequence[Any] | None = None,
    ) -> list[Any]:
        """Paste generated faces back into their frames.

        Args:
            gen_faces: Generated faces.
            frames: Original frames, one per face.
            boxes: Face rectangles in aligned-face space.
            mats: Affine matrices.
            coffs: Colour offsets; when given (and v2 is enabled) the ROI paste-back is used, otherwise the legacy
                whole-rectangle restorer.

        Returns:
            One restored frame per face.
        """
        import torchvision
        from torchvision import transforms

        restored = []
        for i, face in enumerate(gen_faces):
            if coffs is not None and self.paste_v2_enabled:
                restored.append(self.paste_v2(face, frames[i], boxes[i], mats[i], coffs[i]))
                continue
            x1, y1, x2, y2 = boxes[i]
            size = (int(y2 - y1), int(x2 - x1))
            resized = torchvision.transforms.functional.resize(
                face, size=size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
            )
            restored.append(self.image_processor.restorer.restore_img(frames[i], resized, mats[i]))
        return restored
