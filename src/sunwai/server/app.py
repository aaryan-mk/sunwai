"""Application factory for the SunwAI HTTP API.

``create_app`` is the composition root: it loads typed settings, exports them to the flat variables the engine
reads, builds the services and returns the FastAPI application. Run it with
``uvicorn --factory sunwai.server.app:create_app``.
"""

from __future__ import annotations

import hmac
import logging
import os
from collections.abc import Awaitable, Callable, Mapping

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sunwai.core.config import Settings
from sunwai.core.legacy_env import export_to_environ
from sunwai.core.logging import configure_logging
from sunwai.server.config import ServerConfig
from sunwai.server.container import Container, build_container
from sunwai.server.errors import ServiceError
from sunwai.server.legacy_namespace import LegacyNamespace
from sunwai.server.routers import health, jobs, languages, media, segments, web

log = logging.getLogger(__name__)

MUTATING_METHODS = ("POST", "PUT", "PATCH", "DELETE")
Next = Callable[[Request], Awaitable[Response]]


def _install_token_guard(app: FastAPI, token: str) -> None:
    """Require the shared secret on mutating ``/api/`` calls when ``token`` is set (off when empty)."""

    @app.middleware("http")
    async def token_guard(request: Request, call_next: Next) -> Response:
        if token and request.method in MUTATING_METHODS and request.url.path.startswith("/api/"):
            got = request.headers.get("x-dub-token") or request.query_params.get("token") or ""
            if not hmac.compare_digest(got.encode("utf-8", "replace"), token.encode("utf-8")):
                return JSONResponse({"detail": "missing or wrong token"}, status_code=401)
        return await call_next(request)


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(500)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return JSONResponse({"detail": "internal error"}, status_code=500)


def _install_lifecycle(app: FastAPI, container: Container) -> None:
    @app.on_event("startup")
    async def startup() -> None:
        await container.jobs.startup()
        container.focus.restore()

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await container.job_runner.shutdown()


def _install_control_plane(app: FastAPI, container: Container) -> None:
    """Register the action/revision/chat/event routes (off unless ``DUB_ACTIONS``); failure never stops the server."""
    try:
        from sunwai.server.routers import control

        control.register(app, LegacyNamespace(container, app))  # type: ignore[no-untyped-call]  # control.py is not yet typed
    except Exception:
        log.exception("control plane not registered")


def build_app(config: ServerConfig, environ: Mapping[str, str] | None = None) -> FastAPI:
    """Wire the services around ``config`` and return the application.

    Args:
        config: Server configuration.
        environ: Environment used to locate dependency services; defaults to ``os.environ``.

    Returns:
        A new FastAPI application. Startup hooks (orphan sweep, reconciliation) run when the app starts serving.
    """
    container = build_container(config, os.environ if environ is None else environ)
    app = FastAPI(title="Dub Pipeline API")
    app.state.container = container
    _install_token_guard(app, config.token)
    # added AFTER the token guard so CORS is the outermost layer (401s carry CORS headers too)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    _install_error_handlers(app)
    _install_lifecycle(app, container)
    web.mount_static(app, config.web_dir)
    for router in (jobs.router, segments.router, media.router, languages.router, web.router, health.router):
        app.include_router(router)
    _install_control_plane(app, container)
    return app


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the API application from typed settings.

    Args:
        settings: Settings to apply. Defaults to ``Settings()`` (environment and ``.env``).

    Returns:
        The configured FastAPI application.
    """
    configure_logging()
    effective = settings or Settings()
    written = export_to_environ(effective)
    log.info("settings applied (%d flat variables written)", len(written))
    return build_app(ServerConfig.from_environ(os.environ))
