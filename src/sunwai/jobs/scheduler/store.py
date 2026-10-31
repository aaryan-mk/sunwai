"""Filesystem access for one job directory: atomic JSON, chunk plan and per-chunk output state."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

from sunwai.jobs.scheduler.ordering import Chunk

LOCK = threading.RLock()  # serialises every read-modify-write of tracks.json / source-span files in this process


def read_json(path: Path, default: Any = None) -> Any:
    """Read a JSON file, returning ``default`` (``{}`` when None) if it is missing or unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {} if default is None else default


def atomic_json(path: Path, obj: Any) -> None:
    """Write ``obj`` as JSON via a temp file and ``os.replace`` so readers never see a partial file."""
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}_{threading.get_ident()}")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def track_map(state: Any) -> dict[str, Any]:
    """Return the ``tracks`` mapping of a parsed ``tracks.json`` (empty when absent or malformed)."""
    tracks = state.get("tracks") if isinstance(state, dict) else None
    return tracks if isinstance(tracks, dict) else {}


def job_chunks(job_dir: Path) -> list[Chunk]:
    """Return the job's planned chunks sorted by index; malformed entries are skipped."""
    plan = read_json(Path(job_dir) / "plan.json")
    out: list[Chunk] = []
    for c in plan.get("chunks", []) if isinstance(plan, dict) else []:
        try:
            out.append({"idx": int(c["idx"]), "start": float(c["start"]), "dur": float(c["dur"])})
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(out, key=lambda c: c["idx"])


def chunk_dir(job_dir: Path, lang: str, idx: int) -> Path:
    """Return the output directory of one chunk of one language."""
    return Path(job_dir) / lang / f"c{idx:04d}"


def chunk_done(job_dir: Path, lang: str, idx: int) -> bool:
    """Return True when the chunk reports ``done`` and its dubbed video is non-empty."""
    d = chunk_dir(job_dir, lang, idx)
    if read_json(d / "status.json").get("state") != "done":
        return False
    try:
        return (d / f"dub_{lang}.mp4").stat().st_size > 0
    except OSError:
        return False


def file_sha256(path: Path) -> str | None:
    """Return the hex SHA-256 of a file, or None when it cannot be read."""
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None
