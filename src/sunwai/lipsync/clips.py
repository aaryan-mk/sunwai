"""ffmpeg clip helpers of the standalone per-window lip-sync runner."""

from __future__ import annotations

import os
import subprocess

from sunwai.lipsync.media import run


def run_ffmpeg(cmd: list[str]) -> None:
    """Run an ffmpeg command, printing its stderr before re-raising on failure.

    Args:
        cmd: Command and arguments.

    Raises:
        subprocess.CalledProcessError: ffmpeg exited with a non-zero status.
    """
    try:
        run(cmd)
    except subprocess.CalledProcessError as ex:
        print(f"FFmpeg Error: {ex}")
        print(f"Cmd: {cmd}")
        print(f"Stderr: {ex.stderr.decode()}")
        raise


def cut_video_segment(src_video: str, start: float, end: float, out_path: str, target_fps: int = 25) -> bool:
    """Cut a frame-accurate, audio-less segment of a video.

    Args:
        src_video: Source video.
        start: Segment start, seconds.
        end: Segment end, seconds.
        out_path: Destination mp4.
        target_fps: Frame rate to resample to.

    Returns:
        False when the segment is empty, True once it was cut.
    """
    duration = end - start
    if duration <= 0:
        return False
    cmd = ["ffmpeg", "-y", "-i", src_video, "-ss", str(start), "-t", str(duration), "-vf", f"fps={target_fps}"]
    run_ffmpeg([*cmd, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-an", out_path])
    return True


def cut_audio_segment(src_audio: str, start: float, end: float, out_path: str) -> bool:
    """Cut a segment of audio to a 16 kHz mono wav; False when the segment is empty."""
    duration = end - start
    if duration <= 0:
        return False
    run_ffmpeg(
        ["ffmpeg", "-y", "-i", src_audio, "-ss", str(start), "-t", str(duration), "-ar", "16000", "-ac", "1", out_path]
    )
    return True


def concat_clips(clip_paths: list[str], out_path: str, target_fps: int = 25) -> None:
    """Concatenate clips with the concat demuxer (re-encoded); missing clips are skipped with a warning.

    Args:
        clip_paths: Clips in playback order.
        out_path: Destination mp4.
        target_fps: Frame rate of the result.

    Raises:
        RuntimeError: None of the clips exist.
    """
    list_file = out_path + ".txt"
    with open(list_file, "w") as fh:
        for path in clip_paths:
            if os.path.exists(path):
                fh.write(f"file '{os.path.abspath(path)}'\n")
            else:
                print(f"Warning: Missing clip {path}")
    if os.path.getsize(list_file) == 0:
        raise RuntimeError("No clips to concatenate")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-vf", f"fps={target_fps}"]
    run_ffmpeg([*cmd, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-an", out_path])
    os.remove(list_file)


def mux_audio(video_path: str, audio_path: str, out_path: str) -> None:
    """Mux the dub audio into a video without re-encoding the video stream."""
    cmd = ["ffmpeg", "-y", "-i", video_path, "-i", audio_path, "-c:v", "copy", "-c:a", "aac", "-shortest"]
    run_ffmpeg([*cmd, "-map", "0:v", "-map", "1:a", out_path])
