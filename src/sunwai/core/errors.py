"""Domain errors. HTTP status mapping happens only in :mod:`sunwai.server`; these classes carry no HTTP semantics."""

from __future__ import annotations


class SunwaiError(Exception):
    """Base class for all SunwAI errors."""


class ConfigError(SunwaiError):
    """Invalid or missing configuration. Never contains a secret value."""


class ServiceError(SunwaiError):
    """An external service (ASR, MT, TTS, LLM) failed.

    Attributes:
        service: Short service name, e.g. ``"tts"``.
        status: Provider HTTP status when known.
        transient: True when a retry within the deadline may succeed.
    """

    def __init__(self, service: str, message: str, *, status: int | None = None, transient: bool = False) -> None:
        super().__init__(f"{service}: {message}")
        self.service = service
        self.status = status
        self.transient = transient


class DeadlineExceeded(SunwaiError):
    """The time budget for an operation ran out."""


class Cancelled(SunwaiError):
    """The operation was cancelled (for example, a newer revision superseded it)."""


class StageError(SunwaiError):
    """A pipeline stage failed or produced output that failed validation."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(f"stage {stage}: {message}")
        self.stage = stage
