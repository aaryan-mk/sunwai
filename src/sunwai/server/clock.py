"""Wall-clock helpers shared by the API services."""

from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
