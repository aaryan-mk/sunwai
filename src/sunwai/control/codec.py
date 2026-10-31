"""Pure helpers shared by the control-plane modules: timestamps, canonical JSON, hashes and keys."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

Record = dict[str, Any]
"""A JSON-compatible record as stored in the append-only logs."""


def utcnow() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def canon(obj: Any) -> str:
    """Serialise ``obj`` to canonical JSON (sorted keys, no spaces, non-ASCII kept)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(text: str) -> str:
    """Return the SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def seg_key(lang: str, chunk: int | str, seg: str) -> str:
    """Return the store key of one segment: ``<lang>/<chunk>/<segment id>``."""
    return f"{lang}/{int(chunk)}/{seg}"
