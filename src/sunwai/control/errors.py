"""Errors raised by the control plane and mapped to HTTP responses by the router."""

from __future__ import annotations

from typing import Any


class CtlError(Exception):
    """A control-plane failure carrying an HTTP status and a machine-readable code.

    Attributes:
        status: HTTP status code the router should answer with.
        code: Stable machine-readable error code (for example ``stale_revision``).
        detail: Human-readable explanation.
        extra: Additional fields merged into the JSON error body.
    """

    def __init__(self, status: int, code: str, detail: str, **extra: Any) -> None:
        """Create the error.

        Args:
            status: HTTP status code.
            code: Machine-readable error code.
            detail: Human-readable explanation.
            **extra: Additional fields for the response body.
        """
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail
        self.extra = extra
