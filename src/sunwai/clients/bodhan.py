"""Typed clients for the Bodhan ASR, MT and TTS endpoints (hosted API or self-hosted replicas).

The request shapes are identical to those used by the dubbing engine (``sunwai.pipeline.engine``); these classes are the
reusable, injectable form used by new code and tests. They hold one :class:`requests.Session` and translate
failures into :class:`~sunwai.core.errors.ServiceError` without ever including the API key.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import requests

from sunwai.clients import bodhan_endpoints as endpoints
from sunwai.core.errors import ServiceError


def _check(service: str, response: Any) -> Any:
    """Return ``response`` if successful, else raise :class:`ServiceError` (status only, no body, no key)."""
    status = int(response.status_code)
    if status >= 400:
        raise ServiceError(service, f"HTTP {status}", status=status, transient=status == 429 or status >= 500)
    return response


class _Base:
    """Shared session, key and round-robin base selection."""

    service = ""

    def __init__(self, bases: list[str], key: str, session: Any = None, timeout_s: float = 120.0) -> None:
        if not bases:
            raise ServiceError(self.service, "no endpoint configured")
        self._bases = itertools.cycle(bases)
        self._key = key
        self._session = session or requests.Session()
        self._timeout_s = timeout_s

    def _post(self, path: str, **kwargs: Any) -> Any:
        try:
            response = self._session.post(f"{next(self._bases)}{path}", timeout=self._timeout_s, **kwargs)
        except requests.exceptions.RequestException as exc:
            raise ServiceError(self.service, type(exc).__name__, transient=True) from None
        return _check(self.service, response)


class BodhanASRClient(_Base):
    """``indic-transcribe`` speech recognition."""

    service = "asr"

    def transcribe(self, wav_path: Path, language: str | None) -> str:
        """Return the stripped transcript of ``wav_path``."""
        data = {"model": "indic-transcribe", **({"language": language} if language else {})}
        with open(wav_path, "rb") as handle:
            response = self._post(
                "/v1/audio/transcriptions",
                headers=endpoints.auth_headers(self._key),
                files={"file": ("seg.wav", handle, "audio/wav")},
                data=data,
            )
        return str(response.json().get("text", "")).strip()


class BodhanMTClient(_Base):
    """``indic-translate`` machine translation (OpenAI-style chat endpoint)."""

    service = "mt"

    def translate(self, text: str, target_language: str) -> str:
        """Return ``text`` translated into ``target_language``."""
        payload = {
            "model": "indic-translate",
            "messages": [{"role": "user", "content": text}],
            "target_language": target_language,
        }
        response = self._post("/v1/chat/completions", headers=endpoints.auth_headers(self._key, True), json=payload)
        return str(response.json()["choices"][0]["message"]["content"]).strip()


class BodhanTTSClient(_Base):
    """``indic-speak`` speech synthesis."""

    service = "tts"

    def __init__(
        self, bases: list[str], key: str, session: Any = None, timeout_s: float = 120.0, model: str = "indic-speak"
    ) -> None:
        super().__init__(bases, key, session, timeout_s)
        self._model = model

    def synthesize(self, text: str, language: str, voice: str, style: str | None = None) -> bytes:
        """Return WAV bytes for ``text`` spoken in ``voice``."""
        instructions = json.dumps({"lang": language, **({"style": style} if style else {})})
        payload = {"model": self._model, "input": text, "voice": voice, "instructions": instructions}
        response = self._post("/v1/audio/speech", headers=endpoints.auth_headers(self._key, True), json=payload)
        return bytes(response.content)


def build_clients(session: Any = None) -> tuple[BodhanASRClient, BodhanMTClient, BodhanTTSClient]:
    """Create ASR, MT and TTS clients for the active ``BODHAN_MODE`` using keys from the environment."""
    keys = endpoints.load_keys()
    return (
        BodhanASRClient([endpoints.asr_base()], keys["asr"], session),
        BodhanMTClient(endpoints.mt_bases(), keys["mt"], session),
        BodhanTTSClient(endpoints.tts_bases(), keys["tts"], session),
    )
