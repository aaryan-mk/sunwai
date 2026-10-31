"""Dependency map the supervisor probes: which service endpoints exist and which lip-sync workers should run."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass

from sunwai.workers.common import log

DEFAULT_PORTS: dict[str, list[int]] = {
    "asr": [18931],
    "mt": [18933, 18981, 18982, 18983, 18984],
    "it2": [18936],
    "tts": list(range(18990, 19000)),
    "sep": [18952],
}
LIP_MODES = ("proc", "none")
DEFAULT_LIP_WORKERS = ["wA", "wB"]


@dataclass(frozen=True)
class ServiceMap:
    """What the supervisor expects to be running.

    Attributes:
        deps: Service name to its ``host:port`` endpoints.
        lip_mode: ``proc`` (worker processes on this host) or ``none`` (not checkable from here).
        lip_workers: Lip-sync worker ids expected when ``lip_mode`` is ``proc``.
    """

    deps: dict[str, list[str]]
    lip_mode: str
    lip_workers: list[str]


def _valid_endpoints(value: object) -> bool:
    """A non-empty list whose entries all look like ``host:port``."""
    return isinstance(value, list) and bool(value) and all(":" in str(x) for x in value)


def _valid_workers(value: object) -> bool:
    """A non-empty list of non-empty strings."""
    return isinstance(value, list) and bool(value) and all(isinstance(x, str) and x for x in value)


def load_services(environ: Mapping[str, str] | None = None) -> ServiceMap:
    """Build the service map from the environment and an optional services JSON.

    The JSON file (env ``SUP_SERVICES``, else ``DUB_SERVICES``) has the app's ``DUB_SERVICES`` format. Without it the
    endpoints are built from ``DEP_HOST`` (default 127.0.0.1) and the built-in ports; no node is hard-coded. Invalid
    entries are logged and ignored.

    Args:
        environ: Environment to read; defaults to ``os.environ``.

    Returns:
        The service map.
    """
    env = os.environ if environ is None else environ
    host = env.get("DEP_HOST", "127.0.0.1")
    deps = {name: [f"{host}:{port}" for port in ports] for name, ports in DEFAULT_PORTS.items()}
    lip_mode = env.get("LIP_MODE", "proc")
    if lip_mode not in LIP_MODES:
        lip_mode = "proc"
    workers = [w.strip() for w in env.get("LIP_WORKERS", "wA,wB").split(",") if w.strip()] or list(DEFAULT_LIP_WORKERS)
    path = env.get("SUP_SERVICES", env.get("DUB_SERVICES", ""))
    if path:
        try:
            with open(path) as fh:
                config = json.load(fh)
            for name, endpoints in config.items():
                if isinstance(endpoints, list) and name != "lip_workers":
                    if _valid_endpoints(endpoints):
                        deps[name] = [str(x) for x in endpoints]
                    else:
                        log(f"ignoring empty/invalid list for {name}")
            if config.get("lip_mode", lip_mode) in LIP_MODES:
                lip_mode = config.get("lip_mode", lip_mode)
            if _valid_workers(config.get("lip_workers", workers)):
                workers = config.get("lip_workers", workers)
        except Exception as ex:
            log(f"cannot read services file {path}: {ex!r}")
    return ServiceMap(deps, lip_mode, workers)
