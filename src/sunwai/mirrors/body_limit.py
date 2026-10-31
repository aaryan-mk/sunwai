"""ASGI middleware that refuses oversized request bodies before they are parsed or buffered."""

from __future__ import annotations

from collections.abc import Callable

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class BodyTooLarge(BaseException):
    """Raised while receiving a body over the limit.

    It derives from ``BaseException`` so endpoint code with ``except Exception`` cannot swallow it.
    """


class BodyLimitMiddleware:
    """Rejects requests whose body exceeds ``limit_fn(path)`` bytes with HTTP 413.

    The ``Content-Length`` header is checked first; chunked uploads without it are bounded while streaming.

    Args:
        app: The wrapped ASGI application.
        limit_fn: Maps a request path to its maximum body size in bytes.
    """

    def __init__(self, app: ASGIApp, limit_fn: Callable[[str], int]) -> None:
        """Wrap ``app``."""
        self.app = app
        self.limit_fn = limit_fn

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI connection."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self.limit_fn(scope.get("path", ""))

        async def too_large(size: int) -> None:
            body = {"detail": f"request body too large ({size} bytes; limit {limit})"}
            await JSONResponse(body, status_code=413)(scope, receive, send)

        for key, value in scope.get("headers", []):
            if key == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = 0
                if declared > limit:
                    await too_large(declared)
                    return
        seen = 0
        started = False

        async def counting_receive() -> Message:
            nonlocal seen
            message = await receive()
            if message.get("type") == "http.request":
                seen += len(message.get("body", b""))
                if seen > limit:
                    raise BodyTooLarge()
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message.get("type") == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except BodyTooLarge:
            if not started:
                await too_large(seen)
