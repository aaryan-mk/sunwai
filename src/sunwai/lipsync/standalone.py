"""One-shot lip-sync runner for a single job directory (no resident engine).

usage: python -m sunwai.lipsync.standalone --job-dir <dir> --lang <code> [--pad 0.3]

Cuts speech windows, lip-syncs every face run with the stock LatentSync pipeline, falls back to the original clip
where a run or window fails, then concatenates everything and muxes the dub audio. Run with cwd = the LatentSync
checkout. The persistent :class:`sunwai.lipsync.engine.Engine` replaces this for production use.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from typing import Any

from sunwai.lipsync.clips import concat_clips, cut_audio_segment, cut_video_segment, mux_audio
from sunwai.lipsync.latentsync import detect_faces_in_window, get_face_detector, load_pipeline, run_one
from sunwai.lipsync.media import get_video_info
from sunwai.lipsync.runs import split_into_runs
from sunwai.lipsync.windows import create_segments

FPS = 25
GAP_EPSILON_S = 0.01


def _sync_run(pipeline: Any, config: Any, paths: dict[str, str], temp_dir: str, label: str) -> tuple[str, bool]:
    """Lip-sync one run clip; returns the clip to use (synced, or the original on failure) and whether it synced."""
    try:
        run_one(pipeline, config, paths["video"], paths["audio"], paths["synced"], temp_dir)
        if os.path.exists(paths["synced"]) and os.path.getsize(paths["synced"]) > 0:
            return paths["synced"], True
        raise RuntimeError("Output empty")
    except Exception as ex:
        print(f"[{label}] FAILED: {ex}")
        return paths["video"], False


def _interleave_gaps(
    runs: list[tuple[int, int]],
    run_clips: list[str],
    src_video: str,
    start: float,
    end: float,
    n_frames: int,
    prefix: str,
) -> list[str]:
    """Order run clips with original-video clips for the gaps between (and after) them."""
    clips: list[str] = []
    position = 0
    for index, (run_start, run_end) in enumerate(runs):
        if run_start > position:
            gap = f"{prefix}_run_{index}_gap.mp4"
            cut_video_segment(src_video, start + position / FPS, start + run_start / FPS, gap, FPS)
            clips.append(gap)
        clips.append(run_clips[index])
        position = run_end
    if position < n_frames:
        gap = f"{prefix}_final_gap.mp4"
        cut_video_segment(src_video, start + position / FPS, end, gap, FPS)
        clips.append(gap)
    return clips


def process_window(
    pipeline: Any,
    config: Any,
    src_video: str,
    src_audio: str,
    start: float,
    end: float,
    temp_dir: str,
    window_idx: int,
    face_detector: Any,
) -> tuple[str, str, str, float]:
    """Process one speech window: detect faces, lip-sync each face run, keep the rest of the window as is.

    Args:
        pipeline: LatentSync pipeline.
        config: Pipeline config.
        src_video: Source video.
        src_audio: Dub audio.
        start: Window start, seconds.
        end: Window end, seconds.
        temp_dir: Scratch directory.
        window_idx: Window number, used in file names.
        face_detector: LatentSync face detector.

    Returns:
        ``(clip path, status, reason, wall seconds)`` where status is ``synced`` or ``fallback``.
    """
    prefix = os.path.join(temp_dir, f"win_{window_idx}")
    out_path = f"{prefix}_synced.mp4"
    started = time.time()
    status, reason = "fallback", ""
    try:
        cut_video_segment(src_video, start, end, f"{prefix}_input.mp4", FPS)
        cut_audio_segment(src_audio, start, end, f"{prefix}_audio.wav")
        face_mask = detect_faces_in_window(src_video, start, end, face_detector, fps=FPS, sample_step=2)
        if not face_mask:
            raise RuntimeError("No frames processed for detection")
        runs = split_into_runs(face_mask, fps=FPS, min_run_seconds=1.0, bridge_frames=3)
        if not runs:
            raise RuntimeError("No valid face runs found")
        run_clips, synced = [], False
        for index, (run_start, run_end) in enumerate(runs):
            run_prefix = f"{prefix}_run_{index}"
            s_sec, e_sec = start + run_start / FPS, start + run_end / FPS
            paths = {
                "video": f"{run_prefix}_input.mp4",
                "audio": f"{run_prefix}_audio.wav",
                "synced": f"{run_prefix}_synced.mp4",
            }
            print(f"[Window {window_idx}] Processing Run {index}: [{s_sec:.3f}, {e_sec:.3f}]")
            cut_video_segment(src_video, s_sec, e_sec, paths["video"], FPS)
            cut_audio_segment(src_audio, s_sec, e_sec, paths["audio"])
            clip, ok = _sync_run(pipeline, config, paths, temp_dir, f"Window {window_idx}, Run {index}")
            run_clips.append(clip)
            synced = synced or ok
        concat_clips(_interleave_gaps(runs, run_clips, src_video, start, end, len(face_mask), prefix), out_path, FPS)
        status = "synced" if synced else "fallback"
    except Exception as ex:
        reason = str(ex)
        print(f"[Window {window_idx}] FAILED Window Level: {reason}")
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            if os.path.exists(f"{prefix}_input.mp4"):
                shutil.copy(f"{prefix}_input.mp4", out_path)
            else:
                cut_video_segment(src_video, start, end, out_path, FPS)
    return out_path, status, reason, time.time() - started


def _probe_duration(path: str) -> float:
    """Container duration in seconds."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    return float(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip())


def _assemble(windows: list[list[float]], clips: list[str], src_video: str, total: float, temp_dir: str) -> list[str]:
    """Interleave processed window clips with original-video clips for the spans between windows."""
    ordered: list[str] = []
    position = 0.0
    for index, (start, end) in enumerate(windows):
        if start > position + GAP_EPSILON_S:
            gap = os.path.join(temp_dir, f"gap_{index}.mp4")
            print(f"Creating gap clip [{position:.3f}, {start:.3f}]")
            cut_video_segment(src_video, position, start, gap, FPS)
            ordered.append(gap)
        ordered.append(clips[index])
        position = end
    if position < total - GAP_EPSILON_S:
        gap = os.path.join(temp_dir, "gap_final.mp4")
        print(f"Creating final gap clip [{position:.3f}, {total:.3f}]")
        cut_video_segment(src_video, position, total, gap, FPS)
        ordered.append(gap)
    return ordered


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--pad", type=float, default=0.3)
    args = parser.parse_args()
    src_video = os.path.join(args.job_dir, "src.mp4")
    dub_audio = os.path.join(args.job_dir, "dub.wav")
    final_out = os.path.join(args.job_dir, f"dub_{args.lang}_lipsync.mp4")
    for label, path in (("Source video", src_video), ("Dub audio", dub_audio)):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")
    with open(os.path.join(args.job_dir, "segments.json")) as fh:
        segments = json.load(fh)
    fps, width, height = get_video_info(src_video)
    print(f"Source Video: {width}x{height} @ {fps} fps")
    total = _probe_duration(src_video)
    print(f"Total Duration: {total:.3f}s")
    pipeline, config = load_pipeline()
    try:
        face_detector = get_face_detector(device="cuda")
    except Exception as ex:
        print(f"Could not initialize Face Detector: {ex}")
        print("Falling back to simple window processing without face splitting.")
        face_detector = None
    windows = create_segments(segments, args.pad, total)
    print(f"Generated {len(windows)} windows to process.")
    temp_dir = os.path.join(args.job_dir, "temp_lipsync")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    clips = []
    for index, (start, end) in enumerate(windows):
        print(f"Processing Window {index}: [{start:.3f}, {end:.3f}]")
        clip, _, _, _ = process_window(
            pipeline, config, src_video, dub_audio, start, end, temp_dir, index, face_detector
        )
        clips.append(clip)
    print("Concatenating all clips...")
    video_only = os.path.join(temp_dir, "final_video_no_audio.mp4")
    concat_clips(_assemble(windows, clips, src_video, total, temp_dir), video_only, FPS)
    print("Muxing audio...")
    mux_audio(video_only, dub_audio, final_out)
    print(f"Output written to: {final_out}")
    print("ALL DONE")


if __name__ == "__main__":
    main()
