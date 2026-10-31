"""Separation stage: vocal and background stems."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import librosa
import numpy as np
import requests

from sunwai.clients import bodhan_endpoints as bodhan_cfg

log = logging.getLogger(__name__)

from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.stage import StagePolicy
from sunwai.pipeline.stages.base import StageBase

from sunwai.media.audio import save_wav_stereo


# ---------------------------------------------------------------------------
# Stage 9: Separation (Demucs)
# ---------------------------------------------------------------------------
def stage_separate(out_dir: Path):
    """Separate the 44.1 kHz source into vocals and background (GPU service, CPU fallback) and cache the stems."""
    bed = out_dir / "bed.wav"
    voc = out_dir / "vocals.wav"
    if bed.exists() and voc.exists():
        log.info("Stage separate: cached")
        return

    # GPU separation service (sep_server.py): ~1 s vs 40-90 s on contended CPU. Falls back to CPU below.
    _sep = bodhan_cfg.sep_base()
    if _sep and os.environ.get("SEP_GPU", "1") == "1":
        try:
            _r = requests.post(_sep.rstrip("/") + "/separate",
                               json={"in_wav": str(out_dir / "src44k.wav"), "out_dir": str(out_dir)}, timeout=300)
            _r.raise_for_status()
            if bed.exists() and voc.exists():
                log.info("Stage separate: done (GPU service %.1fs)", _r.json().get("secs", -1))
                return
        except Exception as e:
            log.warning("GPU separation failed (%s); falling back to CPU", e)

    import torch
    os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
    torch.set_num_threads(os.cpu_count() or 1)

    import demucs.pretrained as dp
    import demucs.apply as da
    model = dp.get_model("htdemucs")
    model.to("cpu").eval()
    # apply_model signature varies; build signals
    w44 = out_dir / "src44k.wav"
    y, sr = librosa.load(str(w44), sr=44100, mono=False, dtype="float32")
    if y.ndim == 1:
        y = np.stack([y, y])
    # demucs expects (channels, time) batch dim
    sig = torch.from_numpy(y).unsqueeze(0)  # 1 x 2 x T
    with torch.no_grad():
        out = da.apply_model(model, sig, split=True, overlap=0.25, device="cpu")
    stems = out[0].cpu().numpy()  # n_stems x 2 x T
    names = list(model.sources)
    # find vocals index
    try:
        v_idx = names.index("vocals")
    except ValueError:
        v_idx = len(names) - 1
    voc_np = stems[v_idx]                 # 2 x T
    bed_np = stems.sum(axis=0) - voc_np   # sum(non-vocal)

    save_wav_stereo(voc, voc_np.T, 44100)
    save_wav_stereo(bed, bed_np.T, 44100)
    log.info("Stage separate: done")


class SeparateStage(StageBase[None, None]):
    """Separate vocal and background stems."""

    name = "separate"
    policy = StagePolicy(timing_key="separate")

    def run(self, value: None, ctx: ChunkContext) -> None:
        """Separate vocal and background stems."""
        stage_separate(ctx.work_dir)
        return None
