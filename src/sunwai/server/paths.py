"""Validated locations under the job root; every client-supplied name is checked before it touches the disk."""

from __future__ import annotations

import re
from pathlib import Path

LANG_RE = re.compile(r"^[a-z]{2,3}$")
JOB_RE = re.compile(r"^[0-9a-f]{12}$")
SEG_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


class JobPaths:
    """Resolves job, language and chunk directories beneath one job root.

    Attributes:
        root: Absolute job root; holds one directory per job plus ``uploads/``, ``.procs/`` and ``focus.json``.
    """

    def __init__(self, root: Path) -> None:
        """Create the resolver.

        Args:
            root: Job root (resolved to an absolute path).
        """
        self.root = root.resolve()

    @property
    def uploads_dir(self) -> Path:
        """Directory that local source files must live in."""
        return self.root / "uploads"

    @property
    def procs_dir(self) -> Path:
        """Registry of live child processes, used to sweep orphans after a restart."""
        return self.root / ".procs"

    @property
    def focus_file(self) -> Path:
        """Persisted viewer focus read by the lip-sync workers."""
        return self.root / "focus.json"

    def job_dir(self, job_id: str) -> Path:
        """Return the directory of ``job_id`` without validating the identifier.

        Args:
            job_id: A job identifier already known to be well formed.

        Returns:
            ``<root>/<job_id>``.
        """
        return self.root / job_id

    def resolve_job(self, job_id: str) -> Path | None:
        """Resolve a client-supplied job identifier.

        Args:
            job_id: Identifier from the URL.

        Returns:
            The job directory, or ``None`` when the identifier is malformed or escapes the root.
        """
        match = JOB_RE.match(job_id or "")
        if not match:
            return None
        path = (self.root / match.group(0)).resolve()
        return path if path.parent == self.root else None

    def resolve_chunk_dir(self, job_dir: Path, lang: str, chunk_idx: int) -> Path | None:
        """Resolve ``<job>/<lang>/cNNNN``.

        Args:
            job_dir: A job directory.
            lang: Language code from the URL.
            chunk_idx: Chunk index.

        Returns:
            The chunk directory, or ``None`` for a malformed language, a negative index or an escaping path.
        """
        if not LANG_RE.match(lang or "") or chunk_idx < 0:
            return None
        path = (job_dir / lang / f"c{chunk_idx:04d}").resolve()
        return path if path.parent == (job_dir / lang).resolve() else None
