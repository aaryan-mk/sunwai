"""Application factory for the SunwAI HTTP API.

``create_app`` is the single entry point: it loads typed settings, exports them to the flat variables the engine reads,
and returns the FastAPI application. Run it with ``uvicorn --factory sunwai.server.app:create_app``.

The route handlers still live in :mod:`sunwai.server.legacy_app` and are served unchanged; they move into
``routers/`` and ``services/`` module by module without changing this entry point.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from sunwai.core.config import Settings
from sunwai.core.legacy_env import export_to_environ
from sunwai.core.logging import configure_logging

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the API application.

    Args:
        settings: Settings to apply. Defaults to ``Settings()`` (environment and ``.env``).

    Returns:
        The configured FastAPI application. Calling this twice in one process returns the same application, because
        the route module is imported once.
    """
    configure_logging()
    effective = settings or Settings()
    written = export_to_environ(effective)
    log.info("settings applied (%d flat variables written)", len(written))
    from sunwai.server import legacy_app  # imported after the environment is final: it reads it at import time

    return legacy_app.app
