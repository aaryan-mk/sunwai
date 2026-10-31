"""Stage 2 -- audio: extract ``source16k.wav`` (16 kHz mono PCM16) from ``source.mp4``."""

from __future__ import annotations

import logging
import os

from sunwai.planning.artifacts import DERIVED_FROM_WAV, TMP_PREFIX, remove_derived
from sunwai.planning.errors import MediaError, NoAudioError
from sunwai.planning.fileio import remove_file
from sunwai.planning.resources import check_resources
from sunwai.planning.subprocesses import probe, run_ffmpeg
from sunwai.planning.wavio import wav_data_span, wav_ok

log = logging.getLogger("planner")

_PCM16_16K_BYTES_PER_S = 32000.0


def _wav_seconds(path: str) -> float:
    """Real length in seconds of a 16 kHz mono PCM16 wav."""
    return wav_data_span(path)[1] / _PCM16_16K_BYTES_PER_S


def stage_audio(out_dir: str) -> None:
    """Extract the first audio track of ``source.mp4`` into ``source16k.wav`` unless a valid wav is cached.

    Raises:
        MediaError: ``source.mp4`` is unreadable or the extraction was empty/truncated.
        NoAudioError: The video has no audio track.
        ResourceLimitError: The track exceeds the planner's duration/RAM ceiling.
    """
    wav_path = os.path.join(out_dir, "source16k.wav")
    source = os.path.join(out_dir, "source.mp4")
    info = probe(source)
    if info is None:
        raise MediaError("source.mp4 is missing or unreadable")
    if info["n_audio"] < 1:
        raise NoAudioError(
            "this video has no audio track, so there is nothing to dub. "
            "Pick a video with speech (or a range that contains audio)."
        )
    # expected length = the track that gets extracted (0:a:0), not the longest of several audio tracks; unknown -> None
    track_dur = info.get("a0_dur") or None
    if track_dur:
        check_resources(track_dur, "audio")  # the SELECTED track only; also for a cached wav (not bypassable)
    if os.path.exists(wav_path):
        if wav_ok(wav_path, expect_dur=track_dur):
            if not track_dur:  # cached wav, selected track length unknown: ceiling on the wav's real length
                check_resources(_wav_seconds(wav_path), "audio")
            log.info("[2] cached")
            return
        log.warning("[2] source16k.wav failed validation (partial/short): re-extracting")
        remove_file(wav_path)
        remove_derived(out_dir, DERIVED_FROM_WAV)  # products computed from the bad wav are stale (a voice plan is kept)
    log.info(
        "[2] extracting audio (%d video stream(s), %d audio stream(s), %.1fs)",
        info["n_video"],
        info["n_audio"],
        info["duration"],
    )
    _extract(source, wav_path, os.path.join(out_dir, f"{TMP_PREFIX}a{os.getpid()}.wav"), track_dur)


def _extract(source: str, wav_path: str, tmp_wav: str, track_dur: float | None) -> None:
    """Run ffmpeg into ``tmp_wav``, validate it and publish it atomically as ``wav_path``."""
    try:
        run_ffmpeg(
            ["-i", source, "-vn", "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"]
            + ["-af", "aresample=async=1:first_pts=0", "-y", tmp_wav],
            step=2,
        )
        if not wav_ok(tmp_wav, expect_dur=track_dur):
            raise MediaError("audio extraction produced an empty or truncated file")
        if not track_dur:  # selected track's length unknown: ceiling on the real extracted length
            check_resources(_wav_seconds(tmp_wav), "audio")
        os.replace(tmp_wav, wav_path)
    finally:
        remove_file(tmp_wav)
