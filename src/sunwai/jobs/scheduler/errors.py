"""Errors and injected server seams of the add-language scheduler."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class LangSchedError(Exception):
    """A request the scheduler refuses, carrying the HTTP status the API layer should answer with."""

    def __init__(self, status: int, code: str, message: str) -> None:
        """Create the error.

        Args:
            status: HTTP status code.
            code: Stable machine-readable error code.
            message: Human-readable detail.
        """
        super().__init__(message)
        self.status, self.code, self.message = status, code, message

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON error body."""
        return {"status": self.status, "code": self.code, "detail": self.message}


@dataclass
class Hooks:
    """Server seams injected into the scheduler (the module never imports the server).

    Attributes:
        ensure_running: ``job_id`` -> make sure the job's worker is running.
        set_focus: ``(job_id, lang, chunk_idx)`` -> same effect as ``POST /focus``.
        target_codes: Supported target language codes.
        max_targets: Maximum target languages per job.
        merge_lang: ``(job_dir, lang)`` -> add the language to ``job.json``; defaults to the atomic writer.
        busy: True when the server cannot take more dubbing work.
        job_active: ``job_id`` -> True while the job's run task is alive (it may still schedule chunks).
        lipsync_hash_ok: ``(job_dir, lang, idx)`` -> True/False, or None when the lip-sync audio hash is unknown.
    """

    ensure_running: Callable[[str], object] | None = None
    set_focus: Callable[[str, str, int], object] | None = None
    target_codes: Callable[[], set[str]] | None = None
    max_targets: int = 3
    merge_lang: Callable[[Path, str], list[str]] | None = None
    busy: Callable[[], bool] | None = None
    job_active: Callable[[str], bool] | None = None
    lipsync_hash_ok: Callable[[Path, str, int], bool | None] | None = None
