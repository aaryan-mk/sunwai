"""Small filesystem primitives: atomic writes, tolerant JSON reads, file hashing and an advisory file lock."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from types import ModuleType, TracebackType
from typing import IO, Any

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover
    _fcntl = None  # type: ignore[assignment]

fcntl: ModuleType | None = _fcntl
"""The ``fcntl`` module, or ``None`` on platforms without it (locking then degrades to a no-op)."""

_HASH_BLOCK = 1 << 20


def _temp_sibling(path: str | Path) -> Path:
    return Path(str(path) + f".tmp{os.getpid()}_{uuid.uuid4().hex[:6]}")


def write_json_atomic(path: str | Path, obj: Any) -> None:
    """Write ``obj`` as pretty JSON to ``path`` via a temp file and ``os.replace``."""
    tmp = _temp_sibling(path)
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_bytes_atomic(path: str | Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a temp file and ``os.replace``."""
    tmp = _temp_sibling(path)
    tmp.write_bytes(data)
    os.replace(tmp, path)


def read_json(path: str | Path, default: Any = None) -> Any:
    """Read a JSON file, returning ``default`` when it is missing or unparseable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def unlink_quietly(path: str | Path) -> None:
    """Delete a file, ignoring any OS error (missing file, permissions)."""
    try:
        Path(path).unlink()
    except OSError:
        pass


class FileLock:
    """Blocking exclusive ``flock`` on a lock file, usable as a context manager (no-op without ``fcntl``)."""

    def __init__(self, path: str | Path) -> None:
        """Bind the lock to ``path`` (opened on entry)."""
        self.path = str(path)
        self._handle: IO[str] | None = None

    def __enter__(self) -> FileLock:
        """Open the lock file and take the exclusive lock."""
        self._handle = open(self.path, "a+")
        if fcntl:
            fcntl.flock(self._handle, fcntl.LOCK_EX)
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        """Release the lock and close the file."""
        assert self._handle is not None
        if fcntl:
            fcntl.flock(self._handle, fcntl.LOCK_UN)
        self._handle.close()


def lock_is_busy(path: str | Path) -> bool:
    """Probe a lock file without blocking: ``True`` when another holder has it.

    Raises:
        OSError: When the lock file cannot be opened.
        RuntimeError: When the platform has no ``fcntl`` (the caller must treat the state as unknown).
    """
    if fcntl is None:
        raise RuntimeError("file locking unavailable")
    with open(path, "a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle, fcntl.LOCK_UN)
        except OSError:
            return True
    return False
