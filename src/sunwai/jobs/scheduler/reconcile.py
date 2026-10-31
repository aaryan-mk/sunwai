"""Reconcile markers: durable notes that a track needs the client to re-request it."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from sunwai.jobs.scheduler import store


class ReconcileLedger:
    """Records tracks whose state could not be committed, on disk with an in-memory fallback."""

    def __init__(self) -> None:
        """Create an empty ledger."""
        self._memory: dict[tuple[str, str], dict[str, Any]] = {}

    def mark(self, job_dir: Path, lang: str, why: str) -> dict[str, Any]:
        """Record that ``lang`` needs reconciling; the marker file is best effort, memory always holds it."""
        rec = {"lang": lang, "reason": why, "ts": time.time(), "action": "re-request the language to persist the track"}
        self._memory[(str(job_dir), lang)] = rec
        try:
            path = Path(job_dir) / "tracks.reconcile.json"
            data = store.read_json(path)
            data[lang] = rec
            store.atomic_json(path, data)
        except OSError:
            pass
        return rec

    def clear(self, job_dir: Path, lang: str) -> None:
        """Forget the marker for ``lang`` (memory and file)."""
        self._memory.pop((str(job_dir), lang), None)
        path = Path(job_dir) / "tracks.reconcile.json"
        if path.exists():
            data = store.read_json(path)
            if data.pop(lang, None) is not None:
                store.atomic_json(path, data)

    def get(self, job_dir: Path, lang: str) -> dict[str, Any] | None:
        """Return the pending reconcile note for ``lang``, including an uncommitted write-ahead intent."""
        rec = self._memory.get((str(job_dir), lang)) or store.read_json(Path(job_dir) / "tracks.reconcile.json").get(
            lang
        )
        if rec:
            return rec  # type: ignore[no-any-return]
        track = store.track_map(store.read_json(Path(job_dir) / "tracks.json")).get(lang)
        if isinstance(track, dict) and track.get("phase") == "scheduling":
            return {
                "lang": lang,
                "reason": "scheduling intent not committed",
                "action": "re-request the language (recovery rolls it forward)",
            }
        return None


LEDGER = ReconcileLedger()
