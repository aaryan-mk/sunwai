"""yt-dlp adapter: download a URL (or describe a local file) into ``raw.mp4`` and return its metadata."""

from __future__ import annotations

import logging
import os
import re
import shutil
from typing import Any
from urllib.parse import urlparse

from sunwai.planning.cookies import private_cookie_copy
from sunwai.planning.envvars import env_float
from sunwai.planning.errors import YT_BLOCKED_MSG, DownloadBlockedError, MediaError, ResourceLimitError
from sunwai.planning.fileio import norm_path, remove_file
from sunwai.planning.subprocesses import ffprobe_duration

log = logging.getLogger("planner")

Meta = dict[str, Any]

_BOT_RE = re.compile("sign in to confirm you['’‘ʼ`]?re not a bot", re.I)
_YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/|/live/)([A-Za-z0-9_-]{11})")
_PRIMARY_FORMAT = "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b"
_FALLBACK_FORMAT = "b[ext=mp4]/b"


def is_bot_check(text: object) -> bool:
    """True for yt-dlp's exact "Sign in to confirm you're not a bot" diagnostic (apostrophe variants allowed).

    Age-gate, private-video and generic "use --cookies" messages do NOT match.
    """
    return bool(_BOT_RE.search(str(text or "")))


class YdlLog:
    """Forward every yt-dlp message to the planner log and redact the bot-check text on every level.

    The planner reports the bot check as a clean ``PLANNER_ERROR`` instead. yt-dlp's documented level pattern is
    followed: ``[debug] `` messages are debug, other output is info.
    """

    @staticmethod
    def _emit(level: int, msg: object) -> None:
        if is_bot_check(msg):
            log.warning("[1] yt-dlp: YouTube asked for sign-in (bot check)")
        else:
            log.log(level, "[yt-dlp] %s", msg)

    def debug(self, msg: object) -> None:
        """Log a yt-dlp debug line."""
        self._emit(logging.DEBUG if str(msg).startswith("[debug] ") else logging.INFO, msg)

    def info(self, msg: object) -> None:
        """Log a yt-dlp info line."""
        self._emit(logging.INFO, msg)

    def warning(self, msg: object) -> None:
        """Log a yt-dlp warning."""
        self._emit(logging.WARNING, msg)

    def error(self, msg: object) -> None:
        """Log a yt-dlp error."""
        self._emit(logging.ERROR, msg)


def prefetched_youtube_copy(url: str, out_dir: str) -> str | None:
    """Path of a pre-fetched local copy of a YouTube video, or ``None``.

    YouTube blocks cluster IPs (bot check). A copy downloaded elsewhere lives in ``<DUB_ROOT>/uploads/yt_<id>.mp4``
    (see ``fetch_yt_local.sh``); it is used instead of downloading.
    """
    host = (urlparse(url or "").hostname or "").lower()
    if not (host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com")):
        return None
    match = _YOUTUBE_ID_RE.search(url or "")
    if not match:
        return None
    candidate = os.path.join(os.path.dirname(os.path.abspath(out_dir)), "uploads", f"yt_{match.group(1)}.mp4")
    return candidate if os.path.isfile(candidate) and os.path.getsize(candidate) > 0 else None


def _translate_download_error(exc: Exception) -> Exception | None:
    """Map a yt-dlp failure to a clean planner error, or ``None`` when it is not a recognised case."""
    if "download ceiling" in str(exc):
        return ResourceLimitError(str(exc).replace("ERROR: ", ""))
    if is_bot_check(exc):
        return DownloadBlockedError(YT_BLOCKED_MSG)
    return None


def _extract(opts: Meta, url: str) -> Any:
    """Run one yt-dlp download and return its info dict."""
    import yt_dlp

    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=True)


def _download_options(tmp_dir: str) -> Meta:
    """yt-dlp options with the download-length ceiling and bounded retries."""
    max_raw = env_float("PLANNER_MAX_RAW_S", 12 * 3600.0, lo=60.0)

    def too_long(info: Meta, *, incomplete: bool = False) -> str | None:
        duration = info.get("duration")
        if duration and duration > max_raw:
            return f"video is {duration / 3600:.1f} h, longer than the {max_raw / 3600:.0f} h download ceiling"
        return None

    return {
        "format": _PRIMARY_FORMAT,
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(tmp_dir, "raw.%(ext)s"),
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "match_filter": too_long,
        "socket_timeout": 60,
        "retries": 5,
        "logger": YdlLog(),
    }


def _extract_with_fallback(opts: Meta, url: str) -> Any:
    """Download with the primary format; on failure retry once with the best single mp4 stream."""
    try:
        return _extract(opts, url)
    except Exception as first:
        clean = _translate_download_error(first)
        if clean:
            raise clean from None
        log.warning("primary format failed; retrying with best single mp4")
        opts["format"] = _FALLBACK_FORMAT
        try:
            return _extract(opts, url)
        except Exception as second:
            clean = _translate_download_error(second)
            if clean:
                raise clean from None
            raise


def _locate_download(info: Meta, tmp_dir: str, tmp_raw: str) -> str:
    """Path of the file yt-dlp produced (it may not be named ``raw.mp4``)."""
    downloaded = info.get("requested_downloads") or []
    got = norm_path((downloaded[0].get("filepath") if downloaded else None) or tmp_raw)
    if os.path.exists(got):
        return got
    candidates = [
        os.path.join(tmp_dir, n) for n in os.listdir(tmp_dir) if n.startswith("raw.") and not n.endswith(".part")
    ]
    if not candidates:
        raise MediaError("yt_dlp finished but produced no file")
    return candidates[0]


def download_url(url: str, tmp_dir: str, tmp_raw: str) -> Meta:
    """Download ``url`` into ``tmp_raw`` through yt-dlp and return its metadata.

    Raises:
        ResourceLimitError: The video is longer than ``PLANNER_MAX_RAW_S``.
        DownloadBlockedError: YouTube asked for sign-in.
        MediaError: yt-dlp finished without producing a file.
    """
    log.info("[1] downloading via yt_dlp ...")
    opts = _download_options(tmp_dir)
    cookie_copy = private_cookie_copy()
    try:
        if cookie_copy:
            opts["cookiefile"] = cookie_copy
            log.info("[1] using the YouTube cookie file (private copy)")
        info = _extract_with_fallback(opts, url)
    finally:
        if cookie_copy:
            remove_file(cookie_copy)  # the private copy never outlives the download
    if not info:
        raise ResourceLimitError(
            "download refused: video is longer than the download ceiling (PLANNER_MAX_RAW_S) or yt_dlp returned nothing"
        )
    got = _locate_download(info, tmp_dir, tmp_raw)
    if got != tmp_raw:
        os.replace(got, tmp_raw)
    return {k: info.get(k) for k in ("title", "uploader", "duration", "license")} | {"url": url}


def copy_local(url: str, tmp_raw: str) -> Meta:
    """Copy a local input to ``tmp_raw`` and return its metadata (yt-dlp's probe, else ffprobe).

    Raises:
        FileNotFoundError: The input file does not exist.
    """
    log.info("[1] local file -> copy to raw.mp4")
    src = norm_path(url)
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    shutil.copy2(src, tmp_raw)
    try:
        import yt_dlp

        opts = {"quiet": True, "noprogress": True, "no_warnings": True, "simulate": False, "skip_download": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(src, download=False)
        title = info.get("title") or os.path.basename(src)
        return {
            "title": title,
            "uploader": info.get("uploader"),
            "duration": info.get("duration"),
            "license": info.get("license"),
            "url": src,
        }
    except Exception:
        return {
            "title": os.path.basename(src),
            "uploader": None,
            "duration": ffprobe_duration(src),
            "license": None,
            "url": src,
        }
