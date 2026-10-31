"""Validation of the video source a client asks us to dub."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from sunwai.server.errors import ServiceError

ALLOWED_HOSTS = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
        "www.youtu.be",
        "music.youtube.com",
        "studio.youtube.com",
    }
)
MAX_URL_CHARS = 2048


class SourceValidator:
    """Accepts YouTube links and files that live under ``<job root>/uploads``."""

    def __init__(self, uploads_dir: Path) -> None:
        """Create the validator.

        Args:
            uploads_dir: The only directory local source files may come from.
        """
        self._uploads = uploads_dir.resolve()

    def validate(self, url: str) -> str:
        """Return the source in a form safe to hand to the planner.

        Args:
            url: ``http(s)`` YouTube URL, ``file://`` URL, or a path under ``uploads/``.

        Returns:
            The stripped URL, or the absolute path of an existing upload.

        Raises:
            ServiceError: 422 when the source is missing, malformed, from another host or outside ``uploads/``.
        """
        if not url or len(url) > MAX_URL_CHARS:
            raise ServiceError(422, "url is required")
        lowered = url.strip().lower()
        if lowered.startswith("file://"):
            return self._upload_path(lowered[len("file://") :])
        if not lowered.startswith(("http://", "https://")):
            return self._upload_path(url)  # a bare filesystem path must live under uploads/
        try:
            parsed = urlparse(url.strip())
        except Exception:
            raise ServiceError(422, "invalid url") from None
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ServiceError(422, "invalid url")
        if parsed.netloc.split(":")[0].lower() not in ALLOWED_HOSTS:
            raise ServiceError(422, "only youtube.com / youtu.be links are supported")
        return url.strip()

    def _upload_path(self, raw: str) -> str:
        path = Path(raw)
        candidate = path.resolve() if path.is_absolute() else (self._uploads / raw).resolve()
        if not (candidate == self._uploads or self._uploads in candidate.parents):
            raise ServiceError(422, "file path must live under uploads/")
        if not candidate.is_file():
            raise ServiceError(422, f"file not found: {raw}")
        return str(candidate)
