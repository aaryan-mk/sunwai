"""Protocols for the external services the pipeline depends on.

Stages and tests depend on these protocols, never on a concrete HTTP client. Implementations: ``bodhan.py`` (hosted API
or self-hosted replicas) and ``llm.py`` (OpenRouter or Gemini).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ASRClient(Protocol):
    """Speech-to-text."""

    def transcribe(self, wav_path: Path, language: str | None) -> str:
        """Return the transcript of ``wav_path`` (language hint optional)."""
        ...


class MTClient(Protocol):
    """Machine translation into a target language."""

    def translate(self, text: str, target_language: str) -> str:
        """Return ``text`` translated into ``target_language``."""
        ...


class TTSClient(Protocol):
    """Text-to-speech."""

    def synthesize(self, text: str, language: str, voice: str, style: str | None = None) -> bytes:
        """Return PCM16 mono WAV bytes (24 kHz) for ``text`` spoken in ``voice``."""
        ...


class LLMClient(Protocol):
    """Plain-text LLM completion."""

    def complete(self, prompt: str, use_case: str = "fit") -> str:
        """Return the model's text answer for ``prompt``."""
        ...
