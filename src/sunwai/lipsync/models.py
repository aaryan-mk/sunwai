"""Resident LatentSync models for one GPU: UNet, VAE, whisper audio encoder, image processor and DeepCache."""

from __future__ import annotations

import gc
import os
import time
from typing import Any

from sunwai.lipsync.flags import env_enabled

CONFIG_PATH = "configs/unet/stage2_512.yaml"
UNET_CHECKPOINT = "checkpoints/latentsync_unet.pt"
WHISPER_CHECKPOINT = "checkpoints/whisper/tiny.pt"
VAE_REPO = "stabilityai/sd-vae-ft-mse"
DEFAULT_DEEPCACHE_INTERVAL = 3


def gpu_memory_fraction() -> float | None:
    """Per-process CUDA memory cap from env ``LIP_GPU_FRAC``; None when unset or outside ``(0.05, 1)``."""
    fraction = float(os.environ.get("LIP_GPU_FRAC", "") or 0)
    return fraction if 0.05 < fraction < 1.0 else None


class LatentSyncModels:
    """Loads LatentSync once; the process must run with cwd set to the LatentSync checkout.

    Args:
        device: Torch device string for tensors (the weights themselves are placed on ``cuda``).

    Attributes:
        pipe: The ``LipsyncPipeline`` (UNet, VAE, scheduler, whisper audio encoder).
        image_processor: LatentSync ``ImageProcessor`` with the fixed mouth mask.
        resolution: Model input resolution.
        load_seconds: Wall time spent loading.
    """

    def __init__(self, device: str = "cuda") -> None:
        import torch
        from DeepCache import DeepCacheSDHelper
        from latentsync.utils.image_processor import ImageProcessor, load_fixed_mask  # type: ignore[import-not-found]
        from omegaconf import OmegaConf

        started = time.time()
        self.device = torch.device(device)
        self.dtype = torch.float16
        self.cfg = OmegaConf.load(CONFIG_PATH)
        self.pipe = self._load_pipeline()
        self._deepcache = DeepCacheSDHelper(pipe=self.pipe)
        self._deepcache_interval: int | None = None
        mask = load_fixed_mask(self.cfg.data.resolution, self.cfg.data.mask_image_path).float().to("cuda")
        self.image_processor = ImageProcessor(self.cfg.data.resolution, device="cuda", mask_image=mask)
        self.pipe.image_processor = self.image_processor
        self.resolution = self.cfg.data.resolution
        self.set_deepcache(DEFAULT_DEEPCACHE_INTERVAL)
        self._cap_gpu_memory()
        self.load_seconds = time.time() - started

    def _load_pipeline(self) -> Any:
        """Assemble the LatentSync pipeline from the checkpoints in the working directory."""
        from diffusers import AutoencoderKL, DDIMScheduler
        from latentsync.models.unet import UNet3DConditionModel  # type: ignore[import-not-found]
        from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline  # type: ignore[import-not-found]
        from latentsync.whisper.audio2feature import Audio2Feature  # type: ignore[import-not-found]
        from omegaconf import OmegaConf

        data = self.cfg.data
        scheduler = DDIMScheduler.from_pretrained("configs")
        audio_encoder = Audio2Feature(
            model_path=WHISPER_CHECKPOINT,
            device="cuda",
            num_frames=data.num_frames,
            audio_feat_length=data.audio_feat_length,
        )
        vae = AutoencoderKL.from_pretrained(VAE_REPO, torch_dtype=self.dtype)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(self.cfg.model), UNET_CHECKPOINT, device="cpu"
        )
        pipe = LipsyncPipeline(
            vae=vae, audio_encoder=audio_encoder, unet=unet.to(dtype=self.dtype), scheduler=scheduler
        ).to("cuda")
        pipe.unet.eval()
        return pipe

    @staticmethod
    def _cap_gpu_memory() -> None:
        """Apply the optional per-process memory cap for workers that share a GPU."""
        import torch

        try:
            fraction = gpu_memory_fraction()
            if fraction is not None:
                torch.cuda.set_per_process_memory_fraction(fraction)
        except Exception as ex:
            print(f"[engine] LIP_GPU_FRAC ignored: {ex}", flush=True)

    @property
    def audio_encoder(self) -> Any:
        """The whisper audio encoder inside the pipeline."""
        return self.pipe.audio_encoder

    def set_deepcache(self, interval: int) -> None:
        """Enable DeepCache with the given interval; 0 or 1 disables it. No-op when unchanged."""
        interval = int(interval)
        if interval == self._deepcache_interval:
            return
        if self._deepcache_interval:
            self._deepcache.disable()
        if interval > 1:
            self._deepcache.set_params(cache_interval=interval, cache_branch_id=0)
            self._deepcache.enable()
        self._deepcache_interval = interval

    @staticmethod
    def release_cache() -> None:
        """Give cached GPU blocks back after a chunk (workers share GPUs). Kill switch: env ``LIP_EMPTY_CACHE=0``."""
        import torch

        if not env_enabled("LIP_EMPTY_CACHE"):
            return
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as ex:
            print(f"[engine] release_cache failed: {ex}", flush=True)


def cuda_sync() -> None:
    """Block until the current CUDA stream has finished its queued work."""
    import torch

    torch.cuda.current_stream().synchronize()
