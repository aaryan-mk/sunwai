"""The language catalogue (supported target and source languages)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sunwai.server.repository import JobRepository


class LanguageCatalog:
    """Reads ``languages.json``; a missing file means an empty catalogue."""

    def __init__(self, path: Path, repo: JobRepository) -> None:
        """Create the catalogue.

        Args:
            path: ``languages.json`` location.
            repo: Storage used to read the file.
        """
        self._path = path
        self._repo = repo

    def config(self) -> dict[str, Any]:
        """Return the catalogue, ``{"targets": [], "sources": []}`` when the file is missing."""
        if self._path.exists():
            return self._repo.read_json(self._path)
        return {"targets": [], "sources": []}

    def target_codes(self) -> set[str]:
        """Return the codes of all supported target languages."""
        return {target["code"] for target in self.config().get("targets", [])}
