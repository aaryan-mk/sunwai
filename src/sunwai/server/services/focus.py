"""Viewer focus: which chunk, in which language, the viewer is about to watch."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from sunwai.server.errors import ServiceError
from sunwai.server.paths import LANG_RE, JobPaths
from sunwai.server.repository import JobRepository
from sunwai.server.scheduling import FocusRegistry
from sunwai.server.schemas import FocusRequest, FocusResponse

log = logging.getLogger(__name__)

STALE_FOCUS_FILE_S = 900


class FocusService:
    """Records focus reports; drives gate ordering and is persisted atomically for the lip-sync workers."""

    def __init__(self, registry: FocusRegistry, repo: JobRepository, paths: JobPaths) -> None:
        """Create the service.

        Args:
            registry: In-memory focus table.
            repo: Job storage.
            paths: Directory resolver (also locates ``focus.json``).
        """
        self._registry = registry
        self._repo = repo
        self._paths = paths

    def restore(self) -> None:
        """At startup, continue the persisted generation counter and drop a stale ``focus.json``."""
        path = self._paths.focus_file
        try:
            saved = self._repo.read_json(path)
            self._registry.generation = int(saved.get("gen", 0))
            if time.time() - float(saved.get("ts", 0)) > STALE_FOCUS_FILE_S:
                path.unlink()
        except Exception:
            pass

    def set_focus(self, job_id: str, body: FocusRequest) -> FocusResponse:
        """Validate and record a focus report.

        Args:
            job_id: Job being watched.
            body: Language and chunk in view.

        Returns:
            ``ok`` and the new focus generation.

        Raises:
            ServiceError: 404 for an unknown job, 422 for a language or chunk that is not part of it.
        """
        job_dir = self._paths.resolve_job(job_id)
        if job_dir is None or not (job_dir / "job.json").is_file():
            raise ServiceError(404, "job not found")
        lang = (body.lang or "").strip().lower()
        if not LANG_RE.match(lang) or lang not in self._repo.read_active_meta(job_dir).get("langs", []):
            raise ServiceError(422, "language is not part of this job")
        if int(body.chunk) not in self._chunk_indexes(job_dir):
            raise ServiceError(422, "unknown chunk")
        timestamp = self._registry.record(job_id, lang, int(body.chunk))
        self._persist(job_id, lang, int(body.chunk), timestamp)
        return FocusResponse(ok=True, gen=self._registry.generation)

    def _chunk_indexes(self, job_dir: Path) -> set[int]:
        plan = self._repo.read_plan(job_dir)
        try:
            return {int(c["idx"]) for c in plan.get("chunks", [])}
        except Exception:
            return set()

    def _persist(self, job_id: str, lang: str, chunk: int, timestamp: float) -> None:
        record = {
            "job": job_id,
            "lang": lang,
            "chunk": chunk,
            "ts": timestamp,
            "gen": self._registry.generation,
            "server_pid": os.getpid(),
        }
        try:
            self._repo.write_json(self._paths.focus_file, record, indent=None)
        except OSError as exc:
            log.warning("[focus] persist failed %r", exc)
