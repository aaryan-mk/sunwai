"""ROI paste-back of the generated mouth region onto the original frames, with seam colour matching.

Paste-back v2 composites only the generated lower-face region (feathered, colour-offset at the seam) onto the
ORIGINAL frame. The older path replaced the whole aligned face rectangle with a resampled 512 px copy, which softened
the face and left a halo outside the mouth.
"""

from __future__ import annotations

from typing import Any

import kornia
import numpy as np
import torch
import torch.nn.functional as TF
import torchvision
from torchvision import transforms


class PasteBack:
    """Restores generated faces into frames.

    Args:
        dev: Torch device the generated faces live on.
        image_processor: LatentSync ``ImageProcessor`` (provides ``mask_image`` and the legacy ``restorer``).
        paste_v2: Use the ROI paste-back; False falls back to the whole-rectangle restorer.
    """

    def __init__(self, dev: Any, image_processor: Any, paste_v2: bool = True) -> None:
        self.dev = dev
        self.ip = image_processor
        self.paste_v2_enabled = paste_v2

    def _paste_consts(self):
        if getattr(self, "_pc", None) is None:
            G = (1 - self.ip.mask_image[0:1]).to(self.dev).float()[None]           # [1,1,R,R]; 1 = region the model generates
            Gb = (G > 0.5).float()
            er = -TF.max_pool2d(-Gb, 33, 1, 16)                                    # erode ~16 px
            band = (Gb - er).clamp(0, 1)[0]                                        # seam band inside the generated region
            Gf = kornia.filters.gaussian_blur2d(-TF.max_pool2d(-Gb, 9, 1, 4), (25, 25), (7.0, 7.0))[0, 0]
            self._pc = (band, Gf)
        return self._pc

    @torch.no_grad()
    def color_offsets(self, gen_faces, faces_u8):
        """per-frame per-channel mean offset (original - generated) at the seam band, in the [-1,1] RGB face space (both inputs are RGB).
        only SKIN pixels of the original count (YCrCb skin test) and only where generated and original already agree
        (|diff| < 0.25), so hair / ear / neck / background in the band can no longer tint the whole mouth patch; clamped to +-0.06
        (~8 levels); fewer than 300 valid pixels -> no offset. The offset is only ever applied inside the feathered mouth mask (paste_v2)."""
        band, _ = self._paste_consts(); offs = []
        for g, r in zip(gen_faces, faces_u8):
            r8 = r.to(self.dev).float()
            Y = 0.299 * r8[0] + 0.587 * r8[1] + 0.114 * r8[2]
            Cr = (r8[0] - Y) * 0.713 + 128; Cb = (r8[2] - Y) * 0.564 + 128
            skin = ((Cr >= 133) & (Cr <= 173) & (Cb >= 77) & (Cb <= 127)).float()[None]
            ref = r8 / 255 * 2 - 1; d = ref - g.float()
            wgt = band * skin * (d.abs().max(0, keepdim=True)[0] < 0.25).float()
            cnt = wgt.sum()
            mean = (d * wgt).sum((1, 2)) / cnt.clamp_min(1)         # stays on the device (no CUDA sync per face)
            offs.append(torch.where(cnt >= 300, mean.clamp(-0.06, 0.06), torch.zeros_like(mean)))
        return offs

    @torch.no_grad()
    def paste_v2(self, face, frame, box, M, off):
        """ROI version: warps/composites only the destination bounding box of the face rectangle (not the full frame): ~100x less GPU memory at 1080p/4K."""
        dev = self.dev; _, Gf = self._paste_consts()
        x1, y1, x2, y2 = box; h, w = int(y2 - y1), int(x2 - x1); H, W = frame.shape[:2]
        Mt = M if not isinstance(M, np.ndarray) else torch.from_numpy(M).to(dev)[None]
        inv = kornia.geometry.transform.invert_affine_transform(Mt.to(torch.float32))          # aligned-face coords -> frame coords, [1,2,3]
        A = inv[0].cpu().numpy().astype(np.float64)
        pts = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float64) @ A.T          # corners of the pasted rectangle in frame space
        rx0, ry0 = max(0, int(np.floor(pts[:, 0].min())) - 3), max(0, int(np.floor(pts[:, 1].min())) - 3)
        rx1, ry1 = min(W, int(np.ceil(pts[:, 0].max())) + 4), min(H, int(np.ceil(pts[:, 1].max())) + 4)
        if rx1 <= rx0 or ry1 <= ry0:
            return np.ascontiguousarray(frame).copy()
        g = (face.float() + off[:, None, None]).clamp(-1, 1)
        gi = TF.interpolate(g[None], size=(h, w), mode="bicubic", align_corners=False)
        mi = TF.interpolate(Gf[None, None], size=(h, w), mode="bilinear", align_corners=False)
        inv_roi = inv.clone(); inv_roi[0, 0, 2] -= rx0; inv_roi[0, 1, 2] -= ry0                # same transform, ROI-local destination coordinates
        rh, rw = ry1 - ry0, rx1 - rx0
        gw = kornia.geometry.transform.warp_affine(gi, inv_roi, (rh, rw), padding_mode="zeros")[0]
        mw = kornia.geometry.transform.warp_affine(mi, inv_roi, (rh, rw), padding_mode="zeros")[0]
        fr = torch.from_numpy(np.ascontiguousarray(frame[ry0:ry1, rx0:rx1])).to(dev).float().permute(2, 0, 1)
        o = mw * ((gw / 2 + 0.5).clamp(0, 1) * 255) + (1 - mw) * fr
        res = np.ascontiguousarray(frame).copy()
        res[ry0:ry1, rx0:rx1] = o.permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        return res

    def restore(self, gen_faces, frames, boxes, mats, coffs=None):
        out = []
        for i, face in enumerate(gen_faces):
            if coffs is not None and self.paste_v2_enabled:
                out.append(self.paste_v2(face, frames[i], boxes[i], mats[i], coffs[i])); continue
            x1, y1, x2, y2 = boxes[i]
            h, w = int(y2 - y1), int(x2 - x1)
            face = torchvision.transforms.functional.resize(face, size=(h, w), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
            out.append(self.ip.restorer.restore_img(frames[i], face, mats[i]))
        return out
