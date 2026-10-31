"""The name table the control plane reads its collaborators from.

``routers/control.py`` and ``sunwai.control.adapters`` were written against the old single-module server and look
things up by name in a dictionary (``ns["resolve_job"]``, ``ns["FOCUS"]``, ...). This adapter serves exactly those
names from the new services so the control plane works unchanged until it is rewritten to take typed dependencies.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from sunwai.server.atomic import read_json
from sunwai.server.container import Container
from sunwai.server.services.segments import find_segment

FOCUS_GENERATION = "_focus_gen"


class LegacyNamespace(dict[str, Any]):
    """A dictionary of the legacy server names; ``_focus_gen`` reads and writes the live focus counter."""

    def __init__(self, container: Container, app: FastAPI) -> None:
        """Populate the table.

        Args:
            container: The wired services.
            app: The application (control hooks register startup handlers on it).
        """
        cfg = container.config
        super().__init__(
            app=app,
            DUB_ROOT=cfg.root,
            MAX_TARGETS=cfg.max_targets,
            ADM_MAX=cfg.max_uncached_jobs,
            MAX_QUEUED_JOBS=cfg.max_queued_jobs,
            ADM=container.admission,
            FOCUS=container.focus_registry.entries,
            JOB_TASKS=container.job_runner.tasks,
            get_lock=container.locks.get,
            resolve_job=container.paths.resolve_job,
            resolve_chunk_dir=container.paths.resolve_chunk_dir,
            read_json=read_json,
            _status_path=lambda out_dir: out_dir / "status.json",
            _load_segments=container.repo.load_segments,
            _save_segments=container.repo.save_segments,
            _find_segment=find_segment,
            _edit_flow=container.segments.edit_flow,
            get_target_codes=container.catalog.target_codes,
            ensure_job_running=container.job_runner.ensure_running,
        )
        self._focus = container.focus_registry

    def __getitem__(self, key: str) -> Any:
        """Return a name, serving the focus counter live."""
        if key == FOCUS_GENERATION:
            return self._focus.generation
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Return a name or ``default``, serving the focus counter live."""
        if key == FOCUS_GENERATION:
            return self._focus.generation
        return super().get(key, default)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a name; assigning the focus counter updates the registry."""
        if key == FOCUS_GENERATION:
            self._focus.generation = int(value)
            return
        super().__setitem__(key, value)
