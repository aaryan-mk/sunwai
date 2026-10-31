"""Decodability checks that decide whether a cached or freshly produced video file can be trusted."""

from __future__ import annotations

import logging
import os
import re
import subprocess

from sunwai.planning.envvars import ffmpeg_timeout
from sunwai.planning.subprocesses import probe, run_cmd

log = logging.getLogger("planner")

_FRAME_RE = re.compile(r"frame=\s*(\d+)")


def _decode_report(stderr: str | None) -> tuple[int, list[str]]:
    """Return ``(last frame count, error lines)`` from an ``ffmpeg -stats`` stderr dump."""
    text = stderr or ""
    frames = _FRAME_RE.findall(text)
    errors = [ln for ln in text.replace("\r", "\n").splitlines() if ln.strip() and "frame=" not in ln]
    return (int(frames[-1]) if frames else 0), errors


def tail_decodes(path: str, secs: float = 3.0, v_dur: float = 0.0, f_dur: float = 0.0) -> str:
    """Decode the last ``secs`` seconds of the first video stream.

    A download cut short after the header (moov at the front) still probes with the full duration but cannot be
    decoded at its end.

    Returns:
        ``"ok"``, ``"bad"`` (decode errors / timeout / nonzero rc) or ``"empty"`` (clean decode but no frame:
        indistinguishable from an audio-only tail when the video stream duration is unknown, so the caller accepts
        it with a warning rather than reject a valid file).
    """
    # -sseof is relative to the container end; when the video stream ends clearly earlier (audio-only tail) seek
    # relative to the VIDEO end instead, or the tail would legitimately hold no frames
    seek = ["-sseof", f"-{secs:g}"]
    if v_dur > 0 and f_dur > v_dur + 1.0:
        seek = ["-ss", f"{max(0.0, v_dur - secs):.3f}"]
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-stats", *seek]
    cmd += ["-i", path, "-map", "0:V:0", "-an", "-f", "null", "-"]
    try:
        rc, _out, err = run_cmd(cmd, ffmpeg_timeout())
    except (subprocess.TimeoutExpired, OSError):
        return "bad"
    frames, errors = _decode_report(err)
    if rc != 0 or errors:
        return "bad"
    return "ok" if frames > 0 else "empty"


def head_decodes(path: str, secs: float = 5.0) -> bool:
    """Bounded decode from the START: proves the first seconds of video decode cleanly.

    Used when the duration is unknown and the EOF-relative tail decode is not possible.
    """
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-stats", "-t", f"{secs:g}"]
    cmd += ["-i", path, "-map", "0:V:0", "-an", "-f", "null", "-"]
    try:
        rc, _out, err = run_cmd(cmd, ffmpeg_timeout())
    except (subprocess.TimeoutExpired, OSError):
        return False
    frames, errors = _decode_report(err)
    return rc == 0 and not errors and frames > 0


def _check_tail(path: str, info: dict[str, float], known_duration: bool) -> tuple[bool, str] | None:
    """Tail check of :func:`check_video`; ``None`` means the file passed."""
    name = os.path.basename(path)
    if not known_duration:
        log.warning(
            "%s: duration unknown, skipping the EOF-relative tail decode; decoding the first seconds instead", name
        )
        if not head_decodes(path):
            return False, "duration unknown and the first seconds do not decode"
        return None
    state = tail_decodes(path, v_dur=info.get("v_dur", 0.0), f_dur=info["duration"])
    if state == "bad":
        return False, "tail does not decode (truncated?)"
    if state == "empty":
        if info.get("v_dur", 0.0) > 0:
            return False, "no frame in the last seconds of the video stream (truncated?)"
        log.warning(
            "%s: no video frame in the last seconds and the video duration is unknown (audio-only tail?); accepting",
            name,
        )
    return None


def check_video(
    path: str, expect_dur: float | None = None, rel_tol: float = 0.02, abs_tol: float = 2.0, tail: bool = True
) -> tuple[bool, str]:
    """Check that a video is readable, long enough and decodable at its end.

    Used for ``raw.mp4`` / ``source.mp4`` so a partial file from a crashed run is never reused.

    Args:
        path: Video file.
        expect_dur: Expected duration in seconds, if known.
        rel_tol: Relative duration tolerance.
        abs_tol: Absolute duration tolerance in seconds.
        tail: Also decode the last seconds.

    Returns:
        ``(ok, reason)``.
    """
    info = probe(path)
    if info is None:
        return False, "unreadable (ffprobe failed or timed out)"
    if info["n_video"] < 1:
        return False, "no video stream"
    known = info["duration"] > 0.1
    if not known:
        # duration unavailable (fragmented MP4 / TS without a header duration): do not reject a decodable file
        log.warning("%s: container reports no duration; accepting on decodability alone", os.path.basename(path))
    elif expect_dur and expect_dur > 0 and info["duration"] < expect_dur - max(abs_tol, rel_tol * expect_dur):
        return False, f"duration {info['duration']:.1f}s < expected {expect_dur:.1f}s (truncated?)"
    if tail:
        failure = _check_tail(path, info, known)
        if failure:
            return failure
    return True, "ok"
