"""Choosing which media file answers a player request."""

from __future__ import annotations

from pathlib import Path

from sunwai.server.errors import ServiceError
from sunwai.server.paths import LANG_RE, JobPaths
from sunwai.server.repository import JobRepository
from sunwai.server.services.artifacts import ArtifactChecks


def _non_empty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def choose_chunk_file(dub: Path, lipsync: Path, want_lipsync: bool) -> Path:
    """Pick the file to serve for a finished chunk.

    The lip-sync render is preferred when wanted, non-empty and newer than the aligned-audio dub. If only the
    lip-sync render exists it is served even when it was not asked for: better than a 404.

    Args:
        dub: Aligned-audio video.
        lipsync: Lip-synced video.
        want_lipsync: False for ``?lip=0`` or a job created with ``lipsync=false``.

    Returns:
        The path to serve (not guaranteed to exist).
    """
    if want_lipsync and _non_empty(lipsync):
        if not dub.exists() or lipsync.stat().st_mtime > dub.stat().st_mtime:
            return lipsync
        return dub
    if (not dub.exists() or dub.stat().st_size == 0) and _non_empty(lipsync):
        return lipsync
    return dub


class MediaService:
    """Resolves chunk media paths with the same readiness rules the status view uses."""

    def __init__(self, repo: JobRepository, paths: JobPaths, artifacts: ArtifactChecks) -> None:
        """Create the service.

        Args:
            repo: Job storage.
            paths: Directory resolver.
            artifacts: Source-readiness check shared with the status view.
        """
        self._repo = repo
        self._paths = paths
        self._artifacts = artifacts

    def original_chunk(self, job_id: str, chunk: int) -> Path:
        """Return the original-video cut of a chunk (``src.mp4`` of any language's chunk directory).

        Raises:
            ServiceError: 404 when the job or the cut does not exist.
        """
        job_dir = self._paths.resolve_job(job_id)
        if not job_dir:
            raise ServiceError(404, "unknown job")
        langs = self._repo.read_job_meta(job_dir).get("langs", [])
        path = self._artifacts.source_file(job_dir, langs, chunk)
        if path is None or not path.exists():
            raise ServiceError(404, "chunk source not found")
        return path

    def dubbed_chunk(self, job_id: str, lang: str, chunk: int, lip: int = 1) -> Path:
        """Return the dubbed video of a finished chunk.

        Args:
            job_id: Job identifier.
            lang: Target language.
            chunk: Chunk index.
            lip: 0 requests the aligned-audio video even when a lip-sync render exists.

        Raises:
            ServiceError: 404 when the job, language or chunk is invalid, or the video is not ready.
        """
        job_dir = self._paths.resolve_job(job_id)
        if not job_dir:
            raise ServiceError(404, "unknown job")
        if not LANG_RE.match(lang):
            raise ServiceError(404, "unknown lang")
        out_dir = self._paths.resolve_chunk_dir(job_dir, lang, chunk)
        if not out_dir:
            raise ServiceError(404, "chunk dir invalid")
        if self._repo.read_status(out_dir).get("state") != "done":
            raise ServiceError(404, "video not ready")  # never serve a partial file of a queued/running/errored chunk
        want = lip != 0 and self._repo.read_job_meta(job_dir).get("lipsync", True) is not False
        path = choose_chunk_file(out_dir / f"dub_{lang}.mp4", out_dir / f"dub_{lang}_lipsync.mp4", want)
        if not path.exists() or path.stat().st_size == 0:
            raise ServiceError(404, "video not ready")
        return path
