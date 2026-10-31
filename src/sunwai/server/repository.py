"""Job state on disk: ``job.json``, ``plan.json``, per-chunk ``status.json`` and ``segments.json``.

Services depend on the small :class:`JobRepository` protocol; :class:`FileJobRepository` is the existing on-disk
format. Writes are atomic, and status files have a single writer (the event loop) per directory.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from sunwai.server.atomic import JsonObject, atomic_write, atomic_write_bytes, atomic_write_json, read_json
from sunwai.server.clock import utcnow
from sunwai.server.paths import JOB_RE, LANG_RE, JobPaths

Segments = list[dict[str, Any]]


class JobRepository(Protocol):
    """Everything the services need from job storage."""

    def read_json(self, path: Path) -> JsonObject:
        """Read any JSON object file, ``{}`` when absent or invalid."""
        ...

    def write_json(self, path: Path, obj: Any, indent: int | None = 2) -> None:
        """Atomically write a JSON file."""
        ...

    def read_job_meta(self, job_dir: Path) -> JsonObject:
        """Return ``job.json`` regardless of content."""
        ...

    def read_active_meta(self, job_dir: Path) -> JsonObject:
        """Return ``job.json`` only when it names languages, else ``{}``."""
        ...

    def read_plan(self, job_dir: Path) -> JsonObject:
        """Return ``plan.json``."""
        ...

    def read_status(self, out_dir: Path) -> JsonObject:
        """Return the ``status.json`` of a job or chunk directory."""
        ...

    def update_status(self, out_dir: Path, **fields: Any) -> JsonObject:
        """Merge ``fields`` into ``status.json``, stamp ``updated`` and return the new status."""
        ...

    def read_plan_signature(self, job_dir: Path) -> str | None:
        """Return the plan signature the chunk directories were built for."""
        ...

    def write_plan_signature(self, job_dir: Path, signature: str | None) -> bool:
        """Persist the plan signature; True when it is durable (or already holds this value)."""
        ...

    def load_segments(self, chunk_dir: Path) -> Segments:
        """Return the chunk's segments, ``[]`` when absent or invalid."""
        ...

    def save_segments(self, chunk_dir: Path, segments: Segments) -> None:
        """Atomically replace the chunk's ``segments.json``."""
        ...

    def write_bytes(self, path: Path, data: bytes) -> None:
        """Atomically write raw bytes."""
        ...

    def list_job_dirs(self) -> list[Path]:
        """Return every well-named job directory, sorted."""
        ...

    def chunk_status_files(self, job_dir: Path) -> Iterator[Path]:
        """Yield every chunk ``status.json`` of a job."""
        ...

    def purge_chunk_dirs(self, job_dir: Path, langs: list[str]) -> int:
        """Delete all chunk directories; return how many could not be removed."""
        ...

    def restore_interrupted_edits(self, job_dir: Path) -> int:
        """Roll back segment edits killed mid-flight; return how many were restored."""
        ...


class FileJobRepository:
    """:class:`JobRepository` over the job directory tree."""

    def __init__(self, paths: JobPaths) -> None:
        """Create the repository.

        Args:
            paths: Resolver for the job root.
        """
        self._paths = paths

    def read_json(self, path: Path) -> JsonObject:
        """Read any JSON object file, ``{}`` when absent or invalid."""
        return read_json(path)

    def write_json(self, path: Path, obj: Any, indent: int | None = 2) -> None:
        """Atomically write a JSON file."""
        atomic_write_json(path, obj, indent=indent)

    def write_bytes(self, path: Path, data: bytes) -> None:
        """Atomically write raw bytes."""
        atomic_write_bytes(path, data)

    def read_job_meta(self, job_dir: Path) -> JsonObject:
        """Return ``job.json`` regardless of content."""
        return read_json(job_dir / "job.json")

    def read_active_meta(self, job_dir: Path) -> JsonObject:
        """Return ``job.json`` only when it names languages, else ``{}``."""
        meta = read_json(job_dir / "job.json")
        return meta if meta.get("langs") else {}

    def read_plan(self, job_dir: Path) -> JsonObject:
        """Return ``plan.json``."""
        return read_json(job_dir / "plan.json")

    def read_status(self, out_dir: Path) -> JsonObject:
        """Return the ``status.json`` of a job or chunk directory."""
        return read_json(out_dir / "status.json")

    def update_status(self, out_dir: Path, **fields: Any) -> JsonObject:
        """Merge ``fields`` into ``status.json``, stamp ``updated`` and return the new status."""
        status = self.read_status(out_dir)
        status.update(fields)
        status["updated"] = utcnow()
        atomic_write_json(out_dir / "status.json", status)
        return status

    def read_plan_signature(self, job_dir: Path) -> str | None:
        """Return the plan signature the chunk directories were built for."""
        try:
            return (job_dir / "plan.sig").read_text() or None
        except OSError:
            return None

    def write_plan_signature(self, job_dir: Path, signature: str | None) -> bool:
        """Persist the plan signature; a failure is harmless only when the file already holds this value."""
        try:
            atomic_write(job_dir / "plan.sig", signature or "")
            return True
        except OSError:
            return self.read_plan_signature(job_dir) == (signature or None)

    def load_segments(self, chunk_dir: Path) -> Segments:
        """Return the chunk's segments, ``[]`` when absent or invalid."""
        path = chunk_dir / "segments.json"
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if isinstance(data, dict) and "segments" in data:
            return list(data["segments"])
        return list(data) if isinstance(data, list) else []

    def save_segments(self, chunk_dir: Path, segments: Segments) -> None:
        """Atomically replace the chunk's ``segments.json``."""
        atomic_write_json(chunk_dir / "segments.json", segments)

    def list_job_dirs(self) -> list[Path]:
        """Return every well-named job directory, sorted.

        Raises:
            OSError: If the job root cannot be listed.
        """
        return sorted(p for p in self._paths.root.iterdir() if p.is_dir() and JOB_RE.match(p.name))

    def chunk_status_files(self, job_dir: Path) -> Iterator[Path]:
        """Yield every chunk ``status.json`` of a job."""
        return job_dir.glob("*/c*/status.json")

    def purge_chunk_dirs(self, job_dir: Path, langs: list[str]) -> int:
        """Delete all chunk directories under the job's language directories.

        Returns:
            The number of directories that could not be removed.
        """
        failed = 0
        for lang in langs:
            if not LANG_RE.match(lang):
                continue
            for chunk_dir in (job_dir / lang).glob("c[0-9]*"):
                if chunk_dir.is_symlink() or not chunk_dir.is_dir():
                    continue
                shutil.rmtree(chunk_dir, ignore_errors=True)
                if chunk_dir.exists():
                    failed += 1
        return failed

    def restore_interrupted_edits(self, job_dir: Path) -> int:
        """Put back ``segments.json.pre_edit`` snapshots left by an edit that was killed mid-flight."""
        restored = 0
        for snapshot in job_dir.glob("*/c*/segments.json.pre_edit"):
            try:
                os.replace(snapshot, snapshot.with_name("segments.json"))
                self.update_status(snapshot.parent, editing=False)
                restored += 1
            except OSError:
                pass
        return restored
