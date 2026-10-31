"""Audio stage: extract the audio representations."""

from __future__ import annotations

import logging
import os
from pathlib import Path


log = logging.getLogger(__name__)

from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.stage import StagePolicy
from sunwai.pipeline.stages.base import StageBase

from sunwai.media.audio import _wav_ok
from sunwai.media.ffmpeg import run_ffmpeg


def stage_audio(out_dir: Path):
    w16 = out_dir / "src16k.wav"
    w44 = out_dir / "src44k.wav"
    src_mp4 = out_dir / "src.mp4"
    if w16.exists() and w44.exists() and _wav_ok(w16) and _wav_ok(w44):
        log.info("Stage audio: cached")
        return

    for dst, args_ in ((w16, ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"]),
                       (w44, ["-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_f32le"])):
        tmp = out_dir / f".tmp{os.getpid()}_{dst.name}"
        try:
            run_ffmpeg(["-i", str(src_mp4)] + args_ + [str(tmp)])
            if not _wav_ok(tmp):
                raise RuntimeError(f"audio extraction produced an invalid {dst.name}")
            os.replace(tmp, dst)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
    log.info("Stage audio: done")


class AudioStage(StageBase[None, None]):
    """Extract the 16 kHz and 44.1 kHz audio representations."""

    name = "audio"
    policy = StagePolicy(timing_key="audio")

    def run(self, value: None, ctx: ChunkContext) -> None:
        """Extract the 16 kHz and 44.1 kHz audio representations."""
        stage_audio(ctx.work_dir)
        return None
