"""The host interface the control plane depends on (implemented by the server, faked in tests)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class ControlDeps(Protocol):
    """Operations the control plane needs from the dubbing host."""

    def job_dir(self, job_id: str) -> Path | None:
        """Return the job directory, or ``None`` for an unknown job."""

    def job_langs(self, job_dir: Path) -> list[str]:
        """Return the target languages of the job."""

    def chunk_ids(self, job_dir: Path) -> list[int]:
        """Return the chunk indices of the job."""

    def segments(self, job_dir: Path, lang: str, chunk: int) -> tuple[list[dict[str, Any]], float]:
        """Return a chunk's live segments and the chunk's start offset in seconds."""

    def all_segments(self, job_dir: Path, lang: str) -> list[dict[str, Any]]:
        """Return every segment of a language, each tagged with its ``chunk``."""

    def audio_hash(self, job_dir: Path, lang: str, chunk: int, text: str) -> str:
        """Return the hash of the line audio for ``text``."""

    async def edit(self, job_id: str, lang: str, chunk: int, seg_id: str, text: str, expect: dict[str, Any]) -> dict[str, Any]:
        """Replace a line's text through the legacy edit flow; returns ``{"segment": ...}``."""

    async def undo(self, job_id: str, lang: str, chunk: int, seg_id: str, expect: dict[str, Any]) -> dict[str, Any]:
        """Undo a line's last edit through the legacy flow; returns ``{"segment": ...}``."""
