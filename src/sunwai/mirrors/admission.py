"""Request limits and admission control of the IndicTrans2 mirror (kept free of torch so it can be tested)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from fastapi import HTTPException


@dataclass(frozen=True)
class Limits:
    """Request limits of the service.

    Attributes:
        max_body: Largest request body in bytes.
        max_pending: In-flight (running + queued) requests before new ones get 503.
        max_chars: Longest accepted text.
        max_new_tokens: Ceiling for the generated tokens per request.
    """

    max_body: int = 2 << 20
    max_pending: int = 32
    max_chars: int = 4000
    max_new_tokens: int = 1024

    @classmethod
    def from_env(cls) -> Limits:
        """Read the ``IT2_MAX_*`` environment variables."""
        return cls(
            max_body=int(os.environ.get("IT2_MAX_BODY", str(2 << 20))),
            max_pending=int(os.environ.get("IT2_MAX_PENDING", "32")),
            max_chars=int(os.environ.get("IT2_MAX_CHARS", "4000")),
            max_new_tokens=int(os.environ.get("IT2_MAX_TOKENS", "1024")),
        )


class Admission:
    """Counts in-flight requests and refuses oversized or excess ones."""

    def __init__(self, limits: Limits) -> None:
        """Create the gate with no pending requests."""
        self.limits = limits
        self.pending = 0

    def check(self, texts: list[str]) -> None:
        """Raise ``HTTPException`` 413 for an over-long text or 503 when the queue is full."""
        for text in texts:
            if len(text) > self.limits.max_chars:
                raise HTTPException(413, f"text too long ({len(text)} chars; limit {self.limits.max_chars})")
        if self.pending >= self.limits.max_pending:
            raise HTTPException(
                503,
                f"it2 service is overloaded ({self.limits.max_pending} requests queued); retry shortly",
                headers={"Retry-After": "2"},
            )

    def release(self) -> None:
        """Free one pending slot."""
        self.pending -= 1
