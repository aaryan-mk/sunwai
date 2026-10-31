"""Plan, chunk and job artefact checks: completion is derived from files on disk, never from marker files."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from sunwai.server.atomic import JsonObject
from sunwai.server.paths import JobPaths
from sunwai.server.repository import JobRepository

Pair = tuple[str, dict[str, Any]]

SOURCE_STABLE_S = 3


def plan_signature(plan: JsonObject) -> str | None:
    """Fingerprint a plan's chunk layout.

    Args:
        plan: Decoded ``plan.json``.

    Returns:
        A canonical string, or ``None`` when there is no usable plan.
    """
    try:
        chunks = plan["chunks"]
        if not isinstance(chunks, list) or not chunks:
            return None
        return json.dumps([[int(c["idx"]), float(c["start"]), float(c["dur"])] for c in chunks])
    except Exception:
        return None


def _chunk_is_sane(chunk: JsonObject) -> bool:
    int(chunk["idx"])
    start, dur = float(chunk["start"]), float(chunk["dur"])
    return math.isfinite(start) and start >= 0 and math.isfinite(dur) and dur > 0


class ArtifactChecks:
    """Answers "is this plan/chunk/job complete?" from the files present."""

    def __init__(self, repo: JobRepository, paths: JobPaths) -> None:
        """Create the checker.

        Args:
            repo: Job storage.
            paths: Directory resolver.
        """
        self._repo = repo
        self._paths = paths

    def plan_valid(self, job_dir: Path, need_source: bool = True) -> bool:
        """True when ``plan.json`` has sane chunks and (optionally) a non-empty ``source.mp4``."""
        chunks = self._repo.read_plan(job_dir).get("chunks")
        if not isinstance(chunks, list) or not chunks:
            return False
        try:
            if not all(_chunk_is_sane(c) for c in chunks):
                return False
        except Exception:
            return False
        if not need_source:
            return True
        try:
            return (job_dir / "source.mp4").stat().st_size > 0
        except OSError:
            return False

    def chunk_done(self, out_dir: Path | None, lang: str) -> bool:
        """True when the chunk's status is ``done`` and its dubbed mp4 is non-empty."""
        if not out_dir or not out_dir.is_dir():
            return False
        if self._repo.read_status(out_dir).get("state") != "done":
            return False
        try:
            return (out_dir / f"dub_{lang}.mp4").stat().st_size > 0
        except OSError:
            return False

    def missing_pairs(self, job_dir: Path, langs: list[str], plan: JsonObject) -> list[Pair]:
        """Return every ``(lang, chunk)`` that still has to be dubbed, chunk-major."""
        missing: list[Pair] = []
        for chunk in plan.get("chunks", []):
            for lang in langs:
                if not self.chunk_done(self._paths.resolve_chunk_dir(job_dir, lang, chunk["idx"]), lang):
                    missing.append((lang, chunk))
        return missing

    def source_file(self, job_dir: Path, langs: list[str], idx: int) -> Path | None:
        """Return the per-chunk original video: the first non-empty ``src.mp4`` (any language) stable for 3 s.

        Used by both readiness (``has_source``) and the media route so they can never disagree.
        """
        for lang in langs:
            chunk_dir = self._paths.resolve_chunk_dir(job_dir, lang, idx)
            try:
                if chunk_dir:
                    stat = (chunk_dir / "src.mp4").stat()
                    if stat.st_size > 0 and time.time() - stat.st_mtime > SOURCE_STABLE_S:
                        return chunk_dir / "src.mp4"
            except OSError:
                pass
        return None

    def job_needs_work(self, job_dir: Path) -> bool:
        """True when the job has no usable plan, an unapplied plan change, or chunks still missing."""
        meta = self._repo.read_active_meta(job_dir)
        if not meta:
            return False
        if not self.plan_valid(job_dir, need_source=False):
            return True
        plan = self._repo.read_plan(job_dir)
        applied = self._repo.read_plan_signature(job_dir)
        if applied and applied != plan_signature(plan):  # plan.json changed but the purge was never applied
            return True
        return bool(self.missing_pairs(job_dir, meta["langs"], plan))
