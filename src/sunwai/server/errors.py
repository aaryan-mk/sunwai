"""Errors raised by services and translated into HTTP responses by the application factory."""

from __future__ import annotations

from collections.abc import Mapping


class ServiceError(Exception):
    """A request-level failure with the HTTP status it maps to.

    Services raise this instead of importing FastAPI; ``app.py`` turns it into ``{"detail": ...}``.

    Attributes:
        status_code: HTTP status to answer with.
        detail: Human-readable message returned under the ``detail`` key.
        headers: Extra response headers (for example ``Retry-After``).
    """

    def __init__(self, status_code: int, detail: str, headers: Mapping[str, str] | None = None) -> None:
        """Create the error.

        Args:
            status_code: HTTP status to answer with.
            detail: Message returned to the client.
            headers: Optional extra response headers.
        """
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.headers = dict(headers) if headers else None
