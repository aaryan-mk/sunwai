"""VAD stage: speech intervals of the chunk."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import librosa

log = logging.getLogger(__name__)

from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.stage import StagePolicy
from sunwai.pipeline.stages.base import StageBase

from sunwai.pipeline.support.storage import atomic_write_json


# ---------------------------------------------------------------------------
# Stage 3: VAD (Silero)
# ---------------------------------------------------------------------------
def stage_vad(out_dir: Path) -> List[Dict[str, float]]:
    vad_path = out_dir / "vad.json"
    if vad_path.exists():
        log.info("Stage vad: cached")
        with open(vad_path, encoding="utf-8") as f:
            return json.load(f)

    from silero_vad import load_silero_vad, get_speech_timestamps
    model = load_silero_vad()
    w16 = out_dir / "src16k.wav"

    import soundfile as sf
    # resource ceiling: a chunk is <= ~75 s; refuse a runaway input before reading it into RAM
    try:
        _max_s = float(os.environ.get("PIPE_MAX_AUDIO_S", "1800"))
        if not (0 < _max_s < float("inf")):
            raise ValueError
    except ValueError:
        _max_s = 1800.0
    _info = sf.info(str(w16))
    if _info.duration > _max_s:
        raise RuntimeError(f"chunk audio is {_info.duration:.0f}s, over the {_max_s:.0f}s ceiling (PIPE_MAX_AUDIO_S); "
                           f"the planner should have cut it into chunks")
    y, sr = sf.read(str(w16), dtype="float32")
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    y = y.reshape(-1)

    # get_speech_timestamps expects a torch tensor in recent versions
    import torch
    wav_t = torch.from_numpy(y)
    timestamps = get_speech_timestamps(
        wav_t, model,
        threshold=0.5,
        min_silence_duration_ms=100,
        min_speech_duration_ms=100,
        sampling_rate=16000,
        speech_pad_ms=0,
    )
    intervals = []
    for t in timestamps:
        s = t["start"] / sr
        e = t["end"] / sr
        intervals.append({"start": float(s), "end": float(e)})

    atomic_write_json(vad_path, intervals, indent=2)
    log.info("Stage vad: %d intervals", len(intervals))
    return intervals


class VadStage(StageBase[None, list[dict[str, float]]]):
    """Detect speech intervals in the chunk."""

    name = "vad"
    policy = StagePolicy(timing_key="vad")

    def run(self, value: None, ctx: ChunkContext) -> list[dict[str, float]]:
        """Detect speech intervals in the chunk."""
        return stage_vad(ctx.work_dir)  # type: ignore[no-any-return]
