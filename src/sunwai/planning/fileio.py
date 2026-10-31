"""Small filesystem and JSON helpers: atomic writes, tolerant reads and path normalisation."""

from __future__ import annotations

import json
import os
from typing import Any


def norm_path(path: str) -> str:
    """Return an absolute, symlink-resolved path so cached-output checks match the files actually written."""
    try:
        return os.path.realpath(os.path.abspath(os.path.expanduser(path)))
    except Exception:
        return os.path.abspath(os.path.expanduser(path))


def is_url(text: str) -> bool:
    """True for ``http://`` and ``https://`` inputs."""
    return text.startswith("http://") or text.startswith("https://")


def atomic_json(path: str, obj: Any, **dump_kwargs: Any) -> None:
    """Write JSON through a temp file in the same directory, re-parse it, then ``os.replace`` it into place.

    A crash therefore never leaves a half-written JSON file under the final name.
    """
    tmp = f"{path}.tmp{os.getpid()}"
    dump_kwargs.setdefault("ensure_ascii", False)
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, **dump_kwargs)
    with open(tmp, encoding="utf-8") as handle:
        json.load(handle)
    os.replace(tmp, path)


def json_or_none(path: str) -> Any:
    """Parsed JSON, or ``None`` when the file is missing or corrupt (a corrupt cache counts as absent)."""
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def load_json(path: str) -> Any:
    """Parsed JSON; raises when the file is missing or corrupt."""
    with open(path) as handle:
        return json.load(handle)


def remove_file(path: str) -> None:
    """Delete a file, ignoring a missing one."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
