"""Endpoint and credential resolution for the Bodhan speech/MT services (ASR, MT, TTS) and local mirrors.

``BODHAN_MODE`` selects where requests go; the request shape (paths, payloads, ``Authorization: Bearer``) is identical:

* ``api`` (default): every service lives at ``BODHAN_API_BASE`` (default ``https://api.bodhan.ai``).
* ``local``: self-hosted OpenAI-compatible replicas at ``ASR_BASE``, ``MT_BASE``/``MT_BASES`` and ``TTS_BASE``
  (defaults: ``127.0.0.1`` on the conventional ports, host overridable with ``DEP_HOST``).

Always local, in both modes (no hosted equivalent): IndicTrans2 (``IT2_BASE``), the separator (``SEP_BASE``) and
LatentSync lip sync. Source-language ID runs in-process.

Keys come from ``BODHAN_API_KEY`` (shared) or ``BODHAN_<ASR|MT|TTS>_KEY``; each may instead be given as a file path in
``<NAME>_FILE``. Key values are never logged or included in exception text.
"""

from __future__ import annotations

import os
from pathlib import Path

API_BASE_DEFAULT = "https://api.bodhan.ai"
LOCAL_ASR_PORT = 18931
LOCAL_MT_PORTS = (18933, 18981, 18982, 18983, 18984)
LOCAL_TTS_PORTS = tuple(range(18990, 19000))
LOCAL_IT2_PORT = 18936
LOCAL_SEP_PORT = 18952
SERVICES = ("asr", "mt", "tts")


class BodhanConfigError(ValueError):
    """Raised for an invalid ``BODHAN_MODE`` or a missing credential. Never carries a secret value."""


def env_nonblank(name: str, default: str) -> str:
    """Return env var ``name`` unless it is unset or only whitespace/commas, in which case return ``default``."""
    value = os.environ.get(name)
    return value if value is not None and value.replace(",", " ").strip() else default


def _local_host() -> str:
    return env_nonblank("DEP_HOST", "127.0.0.1").strip()


def _local(port: int) -> str:
    return f"http://{_local_host()}:{port}"


def _csv(text: str) -> list[str]:
    return [item.strip().rstrip("/") for item in text.split(",") if item.strip()]


def _fanout(name: str, default: int) -> int:
    """Return a positive int from env ``name`` (capped at 64), else ``default``."""
    try:
        value = int(os.environ.get(name, "").strip())
    except ValueError:
        return default
    return min(value, 64) if value >= 1 else default


def mode() -> str:
    """Return ``"api"`` or ``"local"`` from ``BODHAN_MODE`` (default ``"api"``)."""
    value = (os.environ.get("BODHAN_MODE") or "api").strip().lower()
    if value not in ("api", "local"):
        raise BodhanConfigError(f"BODHAN_MODE must be api|local, got {value!r}")
    return value


def api_base() -> str:
    """Return the hosted API base URL without a trailing slash."""
    return env_nonblank("BODHAN_API_BASE", API_BASE_DEFAULT).strip().rstrip("/")


def asr_base() -> str:
    """Return the ASR base URL for the active mode."""
    if mode() == "api":
        return api_base()
    return env_nonblank("ASR_BASE", _local(LOCAL_ASR_PORT)).strip().rstrip("/")


def mt_bases() -> list[str]:
    """Return MT replica base URLs. In ``api`` mode one URL is repeated ``BODHAN_API_MT_FANOUT`` times (default 5).

    The repetition matters: the pipeline sizes its MT worker pool and in-flight caps from ``len(mt_bases())``.
    """
    if mode() == "api":
        return [api_base()] * _fanout("BODHAN_API_MT_FANOUT", 5)
    if os.environ.get("MT_BASE", "").strip():
        default = os.environ["MT_BASE"]
    else:
        default = ",".join(_local(p) for p in LOCAL_MT_PORTS)
    return _csv(env_nonblank("MT_BASES", default))


def tts_bases() -> list[str]:
    """Return TTS replica base URLs (``api`` mode repeats the hosted URL ``BODHAN_API_TTS_FANOUT`` times)."""
    if mode() == "api":
        return [api_base()] * _fanout("BODHAN_API_TTS_FANOUT", 10)
    return _csv(env_nonblank("TTS_BASE", ",".join(_local(p) for p in LOCAL_TTS_PORTS)))


def it2_base() -> str:
    """Return the IndicTrans2 mirror base URL (local in both modes)."""
    return os.environ.get("IT2_BASE", _local(LOCAL_IT2_PORT))


def sep_base() -> str:
    """Return the source-separation service base URL (local in both modes)."""
    return os.environ.get("SEP_BASE", _local(LOCAL_SEP_PORT))


def _secret(name: str) -> str:
    """Return secret ``name`` from the environment or from the file named by ``<name>_FILE`` ('' when unset)."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    path = os.environ.get(f"{name}_FILE", "").strip()
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise BodhanConfigError(f"cannot read {name}_FILE ({type(exc).__name__})") from None


def load_keys() -> dict[str, str]:
    """Return ``{"asr": ..., "mt": ..., "tts": ...}``.

    ``BODHAN_<SERVICE>_KEY`` wins over the shared ``BODHAN_API_KEY``. In ``api`` mode a missing key raises
    :class:`BodhanConfigError`; in ``local`` mode empty keys are allowed (replicas usually do not authenticate).
    """
    shared = _secret("BODHAN_API_KEY")
    keys: dict[str, str] = {}
    for service in SERVICES:
        key = _secret(f"BODHAN_{service.upper()}_KEY") or shared
        if not key and mode() == "api":
            raise BodhanConfigError(f"set BODHAN_API_KEY (or BODHAN_{service.upper()}_KEY) for BODHAN_MODE=api")
        keys[service] = key
    return keys


def auth_headers(key: str, json_body: bool = False) -> dict[str, str]:
    """Return request headers for a Bearer ``key`` (plus a JSON content type when ``json_body``)."""
    headers = {"Authorization": f"Bearer {key}"}
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


def summary() -> dict[str, object]:
    """Return a log-safe description of the active endpoints (no key values)."""
    return {"mode": mode(), "asr": asr_base(), "mt": mt_bases(), "tts": tts_bases()}
