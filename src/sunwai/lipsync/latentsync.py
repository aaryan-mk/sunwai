"""Model side of the standalone per-window runner: pipeline loading, face detection and single-clip inference."""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any

import numpy as np

STEPS = 20
GUIDANCE_SCALE = 1.5
SEED = 1247
DEEPCACHE_INTERVAL = 3


def load_pipeline() -> tuple[Any, Any]:
    """Load the LatentSync pipeline with DeepCache; run with cwd = the LatentSync checkout.

    Returns:
        ``(pipeline, config)``.

    Raises:
        FileNotFoundError: The UNet config is missing.
        NotImplementedError: The config's cross-attention size matches no whisper checkpoint.
    """
    import torch
    from accelerate.utils import set_seed
    from DeepCache import DeepCacheSDHelper
    from diffusers import AutoencoderKL, DDIMScheduler
    from latentsync.models.unet import UNet3DConditionModel  # type: ignore[import-not-found]
    from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline  # type: ignore[import-not-found]
    from latentsync.whisper.audio2feature import Audio2Feature  # type: ignore[import-not-found]
    from omegaconf import OmegaConf

    print("Loading LatentSync Pipeline...")
    config_path = "configs/unet/stage2_512.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = OmegaConf.load(config_path)
    fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if fp16 else torch.float32
    print(f"Using dtype: {dtype}")
    whisper_models = {768: "checkpoints/whisper/small.pt", 384: "checkpoints/whisper/tiny.pt"}
    if config.model.cross_attention_dim not in whisper_models:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")
    audio_encoder = Audio2Feature(
        model_path=whisper_models[config.model.cross_attention_dim],
        device="cuda",
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model), "checkpoints/latentsync_unet.pt", device="cpu"
    )
    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet.to(dtype=dtype),
        scheduler=DDIMScheduler.from_pretrained("configs"),
    ).to("cuda")
    helper = DeepCacheSDHelper(pipe=pipeline)
    helper.set_params(cache_interval=DEEPCACHE_INTERVAL, cache_branch_id=0)
    helper.enable()
    set_seed(SEED)
    print(f"Pipeline Loaded. Seed: {torch.initial_seed()}")
    return pipeline, config


def get_face_detector(device: str = "cuda") -> Any:
    """Build LatentSync's InsightFace-based ``FaceDetector``.

    Raises:
        ImportError: LatentSync is not importable.
    """
    try:
        from latentsync.utils.face_detector import FaceDetector  # type: ignore[import-not-found]
    except ImportError as ex:
        raise ImportError("FaceDetector not available. Ensure latentsync is installed correctly.") from ex
    detector = FaceDetector(device=device)
    print("Face Detector initialized.")
    return detector


def detect_faces_in_window(
    src_video: str, start: float, end: float, detector: Any, fps: int = 25, sample_step: int = 2
) -> list[bool]:
    """Per-frame face mask of ``[start, end)`` at ``fps``; detection runs every ``sample_step`` frames.

    Args:
        src_video: Source video.
        start: Window start, seconds.
        end: Window end, seconds.
        detector: LatentSync ``FaceDetector``: ``bbox, landmarks = detector(rgb)``.
        fps: Frame rate.
        sample_step: Frames between detections (the last result is held in between).

    Returns:
        One boolean per frame; empty for an empty window.
    """
    import cv2

    n = max(0, int(round((end - start) * fps)))
    if n == 0:
        return []
    mask = np.zeros(n, dtype=bool)
    handle, tmp = tempfile.mkstemp(suffix=".mp4", dir=os.environ.get("TMPDIR"))
    os.close(handle)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src_video, "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}"]
    subprocess.run([*cmd, "-r", str(fps), "-an", "-c:v", "libx264", "-preset", "ultrafast", tmp], check=True)
    capture = cv2.VideoCapture(tmp)
    last = False
    for i in range(n):
        ok, frame = capture.read()
        if not ok:
            break
        if i % sample_step == 0:
            try:
                bbox, _ = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                last = bbox is not None
            except Exception:
                last = False
        mask[i] = last
    capture.release()
    os.remove(tmp)
    print(f"  face frames {int(mask.sum())}/{n} in [{start:.2f}, {end:.2f}]")
    return [bool(v) for v in mask]


def run_one(
    pipeline: Any, config: Any, video_path: str, audio_path: str, out_path: str, temp_dir: str = "temp"
) -> None:
    """Run the pipeline on one clip, mirroring LatentSync's ``scripts/inference.py``."""
    import torch

    fp16 = torch.cuda.get_device_capability()[0] > 7
    pipeline(
        video_path=video_path,
        audio_path=audio_path,
        video_out_path=out_path,
        num_frames=config.data.num_frames,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE_SCALE,
        weight_dtype=torch.float16 if fp16 else torch.float32,
        width=config.data.resolution,
        height=config.data.resolution,
        mask_image_path=config.data.mask_image_path,
        temp_dir=os.path.join(temp_dir, "pipe_tmp"),  # the pipeline rmtree's its temp_dir: never share the window dir
    )
