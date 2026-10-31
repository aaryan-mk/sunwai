"""Typed planner failures.

Every :class:`PlannerError` subclass carries a stable ``code`` that the CLI writes to ``planner_error.json`` and the
server shows to the user, so callers never have to parse messages.
"""

from __future__ import annotations

YT_BLOCKED_MSG = (
    "YouTube blocked this download (sign-in required). "
    "Try another link, or ask the operator to refresh the YouTube cookies."
)


class PlannerError(RuntimeError):
    """A failure the planner reports cleanly (``planner_error.json`` + a final ``PLANNER_ERROR`` stderr line + rc 3)."""

    code = "planner_error"


class NoAudioError(PlannerError):
    """The input video has no audio track, so there is nothing to dub."""

    code = "no_audio"


class ResourceLimitError(PlannerError):
    """The input is longer, or needs more RAM, than the configured ceiling."""

    code = "resource_limit"


class MediaError(PlannerError):
    """The media is unreadable, truncated or otherwise unusable."""

    code = "bad_media"


class SubprocessTimeout(PlannerError):
    """An ffmpeg/ffprobe subprocess exceeded its time limit."""

    code = "timeout"


class DownloadBlockedError(PlannerError):
    """The remote site refused the download (YouTube bot check)."""

    code = "download_blocked"
