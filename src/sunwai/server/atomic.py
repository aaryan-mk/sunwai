"""Atomic file publication and tolerant JSON reading: readers never observe a half-written file."""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]


def atomic_write_bytes(path: Path | str, data: bytes) -> None:
    """Write ``data`` to ``path`` through a fsynced temporary file and an atomic rename.

    Args:
        path: Destination file.
        data: Bytes to publish.

    Raises:
        OSError: If the temporary file cannot be written or renamed; the temporary file is removed first.
    """
    target = Path(path)
    tmp = target.with_name(f".{target.name}.tmp{os.getpid()}_{uuid.uuid4().hex[:6]}")
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def atomic_write(path: Path | str, text: str) -> None:
    """Atomically write UTF-8 ``text`` to ``path``.

    Args:
        path: Destination file.
        text: Text to publish.
    """
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path | str, obj: Any, indent: int | None = 2) -> None:
    """Atomically write ``obj`` as JSON.

    Args:
        path: Destination file.
        obj: JSON-serialisable value.
        indent: ``json.dumps`` indent; ``None`` writes a single line.
    """
    atomic_write(path, json.dumps(obj, indent=indent, ensure_ascii=False))


def read_json(path: Path) -> JsonObject:
    """Read a JSON object, treating a missing, unreadable or non-object file as empty.

    Args:
        path: File to read.

    Returns:
        The decoded object, or ``{}``.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}
