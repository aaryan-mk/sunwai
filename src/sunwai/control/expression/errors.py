"""Errors and the kill switch of the expression subsystem."""

from __future__ import annotations

import os

ON_WORDS = ("1", "true", "yes", "on")


class ExprError(Exception):
    """An expression-layer failure that maps onto an HTTP-like status.

    Attributes:
        status: HTTP-style status code (404, 409, 422, 503, ...).
        msg: Human-readable explanation.
    """

    def __init__(self, status: int, msg: str) -> None:
        """Create the error from a status code and a message."""
        super().__init__(msg)
        self.status = status
        self.msg = msg


def enabled() -> bool:
    """Tell whether expression is on: only when ``EXPRESSION_ENABLED`` is truthy (default OFF)."""
    value = (os.environ.get("EXPRESSION_ENABLED") or "").strip().lower()
    return value in ON_WORDS
