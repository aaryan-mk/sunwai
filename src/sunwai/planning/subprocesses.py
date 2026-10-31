"""Bounded subprocess helpers and the ffmpeg/ffprobe wrappers built on them."""

from __future__ import annotations

import json
import logging
import math
import os
import signal
import subprocess
from typing import Any

from sunwai.planning.envvars import ffmpeg_timeout, ffprobe_timeout
from sunwai.planning.errors import SubprocessTimeout

log = logging.getLogger("planner")

MediaInfo = dict[str, Any]


def run_cmd(cmd: list[str], timeout: float) -> tuple[int, str, str]:
    """Run a command with a hard timeout that kills the whole process group.

    A plain ``subprocess.run(timeout=)`` leaves grandchildren holding the pipes open and blocks forever.

    Returns:
        ``(returncode, stdout, stderr)``.

    Raises:
        subprocess.TimeoutExpired: When the command outlived ``timeout``.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        raise
    return proc.returncode, out, err


def run_ffmpeg(args: list[str], step: int = 0, timeout: float | None = None) -> None:
    """Run ``ffmpeg`` with the given arguments.

    Raises:
        SubprocessTimeout: When ffmpeg exceeded the time limit.
        RuntimeError: When ffmpeg exited non-zero (the message carries the stderr tail).
    """
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"] + args
    log.debug("[ffmpeg s%d] %s", step, " ".join(cmd))
    limit = ffmpeg_timeout() if timeout is None else timeout
    try:
        rc, _out, err = run_cmd(cmd, limit)
    except subprocess.TimeoutExpired:
        raise SubprocessTimeout(f"ffmpeg step {step} timed out after {limit:.0f}s (PLANNER_FFMPEG_TIMEOUT)") from None
    if rc != 0:
        tail = (err or "").strip().splitlines()[-10:]
        raise RuntimeError(f"ffmpeg failed (rc={rc}): " + "\n".join(tail))


def _positive_float(value: object) -> float:
    """``float(value)`` when it is finite and positive, else 0.0."""
    try:
        number = float(value)  # type: ignore[arg-type]
        return number if math.isfinite(number) and number > 0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def parse_probe_json(doc: dict[str, Any]) -> MediaInfo:
    """Summarise ``ffprobe -of json`` output into stream counts and durations.

    Attached-picture "video" streams are not counted. ``a0_dur`` is the first audio track, the one
    :func:`sunwai.planning.audio.stage_audio` extracts.
    """
    n_video = n_audio = 0
    v_dur = a_dur = a0_dur = 0.0
    for stream in doc.get("streams", []):
        if stream.get("codec_type") == "video" and not (stream.get("disposition") or {}).get("attached_pic"):
            n_video += 1
            v_dur = max(v_dur, _positive_float(stream.get("duration")))
        elif stream.get("codec_type") == "audio":
            n_audio += 1
            a_dur = max(a_dur, _positive_float(stream.get("duration")))
            if n_audio == 1:
                a0_dur = _positive_float(stream.get("duration"))
    # TS/fragmented files have no container duration: fall back to the longest stream
    duration = _positive_float((doc.get("format") or {}).get("duration")) or max(v_dur, a_dur)
    return {
        "duration": duration,
        "n_video": n_video,
        "n_audio": n_audio,
        "v_dur": v_dur,
        "a_dur": a_dur,
        "a0_dur": a0_dur,
    }


def probe(path: str) -> MediaInfo | None:
    """One ffprobe call (with timeout).

    Returns:
        ``{"duration", "n_video", "n_audio", "v_dur", "a_dur", "a0_dur"}``, or ``None`` when the file is
        missing, unreadable or the probe timed out.
    """
    try:
        if not os.path.exists(path) or os.path.getsize(path) < 64:
            return None
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "stream=codec_type,duration:stream_disposition=attached_pic:format=duration",
            "-of", "json", path,
        ]  # fmt: skip
        rc, out, _err = run_cmd(cmd, ffprobe_timeout())
        if rc != 0:
            return None
        return parse_probe_json(json.loads(out or "{}"))
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return None


def ffprobe_duration(path: str) -> float:
    """Container duration in seconds, or 0.0 when unknown."""
    info = probe(path)
    return info["duration"] if info else 0.0
