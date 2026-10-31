"""Batched LatentSync denoising of aligned face crops.

The UNet works on 16-frame windows. Full windows are batched ``batch`` at a time (windows are independent: same
noise, own audio condition); a trailing partial window is run on its own.
"""

from __future__ import annotations

import math
from typing import Any

from sunwai.lipsync.models import LatentSyncModels

WINDOW_FRAMES = 16
VAE_CHUNK = 32


def plan_window_groups(
    n_frames: int, batch: int, window: int = WINDOW_FRAMES
) -> tuple[list[tuple[int, int]], list[list[int]]]:
    """Split ``n_frames`` into UNet windows and group them into batches.

    Args:
        n_frames: Number of frames to generate.
        batch: Full windows per UNet batch.
        window: Frames per window.

    Returns:
        ``(windows, groups)``: ``windows`` are ``[start, end)`` frame ranges, ``groups`` lists window indices per
        UNet call (full windows batched, each partial window alone).
    """
    windows = [(w * window, min(n_frames, (w + 1) * window)) for w in range(math.ceil(n_frames / window))]
    full = [w for w, (a, b) in enumerate(windows) if b - a == window]
    partial = [w for w, (a, b) in enumerate(windows) if b - a != window]
    groups = [full[i : i + batch] for i in range(0, len(full), batch)] + [[w] for w in partial]
    return windows, groups


class Denoiser:
    """Runs the LatentSync UNet over aligned faces.

    Args:
        models: Resident LatentSync models.
    """

    def __init__(self, models: LatentSyncModels) -> None:
        self.models = models

    def denoise(self, faces_u8: Any, wchunks: list[Any], steps: int, gs: float, batch: int, seed: int) -> list[Any]:
        """Generate lip-synced faces.

        Args:
            faces_u8: ``[n, 3, R, R]`` uint8 CPU tensor of aligned faces.
            wchunks: One whisper audio feature tensor per frame.
            steps: DDIM steps.
            gs: Classifier-free guidance scale (disabled at 1.0 or below).
            batch: Full windows per UNet batch.
            seed: Noise seed.

        Returns:
            One generated ``[3, R, R]`` fp16 CUDA face per input frame.
        """
        import torch

        with torch.no_grad():
            return self._denoise(faces_u8, wchunks, steps, gs, batch, seed)

    def _denoise(self, faces_u8: Any, wchunks: list[Any], steps: int, gs: float, batch: int, seed: int) -> list[Any]:
        """Body of :meth:`denoise`, run without autograd."""
        import torch

        models = self.models
        n = faces_u8.shape[0]
        windows, groups = plan_window_groups(n, batch)
        generator = torch.Generator(device=models.device).manual_seed(seed)
        models.pipe.scheduler.set_timesteps(steps, device=models.device)
        latent_hw = models.resolution // models.pipe.vae_scale_factor
        noise = torch.randn(
            (1, 4, 1, latent_hw, latent_hw), generator=generator, device=models.device, dtype=models.dtype
        )
        results: dict[int, Any] = {}
        for group in groups:
            ids = [i for w in group for i in range(*windows[w])]
            frames_per_window = windows[group[0]][1] - windows[group[0]][0]
            decoded = self._denoise_group(
                faces_u8[ids].to(models.device),
                torch.stack([wchunks[i] for i in ids]),
                (len(group), frames_per_window, latent_hw),
                noise,
                generator,
                gs,
            )
            for k, i in enumerate(ids):
                results[i] = decoded[k]
        return [results[i] for i in range(n)]

    def _denoise_group(
        self, faces: Any, audio: Any, shape: tuple[int, int, int], noise: Any, generator: Any, gs: float
    ) -> Any:
        """Denoise one batch of windows and return the decoded frames ``[B * F, 3, R, R]``."""
        import torch
        import torch.nn.functional as tf
        from einops import rearrange
        from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline  # type: ignore[import-not-found]

        models = self.models
        pipe, dev, dt = models.pipe, models.device, models.dtype
        batch_windows, frames_per_window, latent_hw = shape
        cfg_on = gs > 1.0
        ref_pv, masked_pv, masks = models.image_processor.prepare_masks_and_masked_images(faces, affine_transform=False)
        mask_l = tf.interpolate(masks, size=(latent_hw, latent_hw)).to(dt)
        masked_l = self._vae_encode(masked_pv, generator)
        ref_l = self._vae_encode(ref_pv, generator)
        mask_l, masked_l, ref_l = (
            rearrange(t, "(b f) c h w -> b c f h w", b=batch_windows) for t in (mask_l, masked_l, ref_l)
        )
        aud = audio.to(dev, dtype=dt)
        if cfg_on:
            mask_l, masked_l, ref_l = (torch.cat([t] * 2) for t in (mask_l, masked_l, ref_l))
            aud = torch.cat([torch.zeros_like(aud), aud])
        latents = noise.repeat(batch_windows, 1, frames_per_window, 1, 1) * pipe.scheduler.init_noise_sigma
        for t in pipe.scheduler.timesteps:
            model_in = torch.cat([latents] * 2) if cfg_on else latents
            model_in = pipe.scheduler.scale_model_input(model_in, t)
            model_in = torch.cat([model_in, mask_l, masked_l, ref_l], dim=1)
            pred = pipe.unet(model_in, t, encoder_hidden_states=aud).sample
            if cfg_on:
                uncond, cond = pred.chunk(2)
                pred = uncond + gs * (cond - uncond)
            latents = pipe.scheduler.step(pred, t, latents, eta=0.0).prev_sample
        decoded = self._vae_decode(
            latents.permute(0, 2, 1, 3, 4).reshape(batch_windows * frames_per_window, 4, latent_hw, latent_hw)
        )
        return LipsyncPipeline.paste_surrounding_pixels_back(decoded, ref_pv, 1 - masks, dev, dt)

    def _vae_encode(self, x: Any, generator: Any) -> Any:
        """Encode images to scaled VAE latents, in chunks to bound memory."""
        import torch

        vae = self.models.pipe.vae
        outs = []
        for i in range(0, x.shape[0], VAE_CHUNK):
            z = vae.encode(x[i : i + VAE_CHUNK].to(self.models.dtype)).latent_dist.sample(generator=generator)
            outs.append((z - vae.config.shift_factor) * vae.config.scaling_factor)
        return torch.cat(outs)

    def _vae_decode(self, z: Any) -> Any:
        """Decode scaled latents to images, in chunks to bound memory."""
        import torch

        vae = self.models.pipe.vae
        z = z / vae.config.scaling_factor + vae.config.shift_factor
        return torch.cat([vae.decode(z[i : i + VAE_CHUNK]).sample for i in range(0, z.shape[0], VAE_CHUNK)])
