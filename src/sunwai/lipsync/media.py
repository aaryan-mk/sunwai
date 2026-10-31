"""ffmpeg / ffprobe helpers for the lip-sync engine: probing, decoding, audio cutting and streaming encode."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import IO, Any

import numpy as np
import numpy.typing as npt

FFMPEG_QUIET = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error"]
FALLBACK_VIDEO_INFO = (25.0, 512, 512)


def run(cmd: list[str]) -> subprocess.CompletedProcess[bytes]:
    """Run a command, capturing stdout/stderr and raising on a non-zero exit.

    Args:
        cmd: Command and arguments.

    Returns:
        The completed process.
    """
    return subprocess.run(cmd, check=True, capture_output=True)


def duration_s(path: str) -> float:
    """Return the container duration of a media file in seconds."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path],
        capture_output=True,
        text=True,
    )
    return float(probe.stdout.strip())


def parse_frame_rate(text: str) -> float:
    """Parse an ffprobe ``r_frame_rate`` value such as ``30000/1001`` or ``25``."""
    if "/" in text:
        numerator, denominator = text.split("/")
        return float(numerator) / float(denominator)
    return float(text)


def get_video_info(video_path: str) -> tuple[float, int, int]:
    """Probe the first video stream.

    Args:
        video_path: Path to the video.

    Returns:
        ``(fps, width, height)``; ``(25.0, 512, 512)`` when probing fails.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,width,height",
        "-of",
        "default=noprint_wrappers=1",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        fields = dict(line.split("=", 1) for line in result.stdout.strip().split("\n"))
        return parse_frame_rate(fields["r_frame_rate"]), int(fields["width"]), int(fields["height"])
    except Exception as ex:
        print(f"Error getting video info: {ex}")
        return FALLBACK_VIDEO_INFO


def decode_frames(src: str, fps: int, width: int, height: int) -> npt.NDArray[np.uint8]:
    """Decode a video to an RGB frame array at a fixed frame rate.

    Args:
        src: Video path.
        fps: Frame rate to resample to.
        width: Frame width reported by the probe.
        height: Frame height reported by the probe.

    Returns:
        Array of shape ``[N, H, W, 3]``.
    """
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", src, "-vf", f"fps={fps}"]
    raw = run([*cmd, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]).stdout
    count = len(raw) // (width * height * 3)
    return np.frombuffer(raw, np.uint8)[: count * width * height * 3].reshape(count, height, width, 3)


def read_mono_wav16(src: str) -> npt.NDArray[np.int16]:
    """Extract the audio of ``src`` as 16 kHz mono int16 samples.

    Args:
        src: Media path.

    Returns:
        The decoded samples.
    """
    from scipy.io import wavfile

    scratch = tempfile.mkdtemp(prefix="asd_", dir=os.environ.get("TMPDIR"))
    try:
        wav_path = os.path.join(scratch, "o.wav")
        run([*FFMPEG_QUIET, "-i", src, "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", wav_path])
        samples: npt.NDArray[np.int16] = wavfile.read(wav_path)[1]
        return samples
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def cut_audio_run(dub_wav: str, start_frame: int, n_frames: int, fps: int, offset_ms: float, out_path: str) -> None:
    """Cut the dub audio that drives one run to a 16 kHz mono wav.

    A positive ``offset_ms`` takes the audio features from earlier audio; when that points before the start of the
    dub the audio is delayed with silence instead.

    Args:
        dub_wav: Dub audio of the whole chunk.
        start_frame: First video frame of the run.
        n_frames: Run length in frames.
        fps: Video frame rate.
        offset_ms: Audio offset in milliseconds.
        out_path: Destination wav.
    """
    audio_start = start_frame / fps - offset_ms / 1000.0
    length = f"{n_frames / fps:.3f}"
    tail = ["-ar", "16000", "-ac", "1", out_path]
    if audio_start < 0:
        delay = f"adelay={int(-audio_start * 1000)}:all=1"
        run([*FFMPEG_QUIET, "-i", dub_wav, "-af", delay, "-ss", "0", "-t", length, *tail])
    else:
        run([*FFMPEG_QUIET, "-ss", f"{audio_start:.3f}", "-t", length, "-i", dub_wav, *tail])


class VideoEncoder:
    """Streams raw RGB frames into an ffmpeg process that encodes H.264 and muxes the dub audio.

    Args:
        width: Frame width.
        height: Frame height.
        fps: Frame rate.
        dub_wav: Audio muxed alongside the video.
        crf: x264 constant rate factor.
        out_path: Destination mp4 (written directly; callers publish it atomically afterwards).
    """

    def __init__(self, width: int, height: int, fps: int, dub_wav: str, crf: int, out_path: str) -> None:
        self.out_path = out_path
        cmd = [
            *FFMPEG_QUIET,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-i",
            dub_wav,
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            "-f",
            "mp4",
            out_path,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.frames_written = 0

    def write(self, frames: npt.NDArray[np.uint8]) -> None:
        """Append frames (``[n, H, W, 3]`` uint8) to the stream."""
        stdin: IO[Any] = self._proc.stdin  # type: ignore[assignment]
        stdin.write(frames.data)
        self.frames_written += len(frames)

    def finish(self, tail: npt.NDArray[np.uint8] | None = None) -> None:
        """Write the remaining frames, close the stream and wait for ffmpeg.

        Kills the encoder and removes the partial file on any failure.

        Args:
            tail: Final frames still to be written, if any.

        Raises:
            RuntimeError: ffmpeg exited with a non-zero status.
        """
        try:
            if tail is not None and len(tail):
                self.write(tail)
            stdin: IO[Any] = self._proc.stdin  # type: ignore[assignment]
            stdin.close()
            if self._proc.wait() != 0:
                raise RuntimeError("ffmpeg encode failed")
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        """Kill the encoder and delete the partial output file."""
        try:
            stdin: IO[Any] = self._proc.stdin  # type: ignore[assignment]
            stdin.close()
        except Exception:
            pass
        self._proc.kill()
        self._proc.wait()
        try:
            os.remove(self.out_path)
        except OSError:
            pass
