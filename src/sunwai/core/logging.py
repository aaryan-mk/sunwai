"""Logging setup with structured context (job, language, chunk, stage) and no secrets."""

from __future__ import annotations

import logging
from typing import Any

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once (idempotent)."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=_FORMAT, datefmt="%H:%M:%S")
    else:
        root.setLevel(level)


class ContextAdapter(logging.LoggerAdapter):  # type: ignore[type-arg]
    """Prefix every message with ``key=value`` context pairs, e.g. ``job=ab12 lang=hi chunk=3``."""

    def process(self, msg: Any, kwargs: Any) -> tuple[Any, Any]:
        """Prepend the bound context to ``msg``."""
        ctx = " ".join(f"{k}={v}" for k, v in (self.extra or {}).items())
        return (f"[{ctx}] {msg}" if ctx else msg), kwargs


def bind(logger: logging.Logger, **context: object) -> ContextAdapter:
    """Return ``logger`` with ``context`` attached to every record."""
    return ContextAdapter(logger, context)
