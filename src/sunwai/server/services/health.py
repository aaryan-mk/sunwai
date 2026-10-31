"""Liveness and dependency probing: this process plus every service the pipeline needs."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeGuard

from sunwai.server.clock import utcnow
from sunwai.server.config import ServerConfig
from sunwai.server.scheduling import JobAdmission
from sunwai.server.schemas import AdmissionInfo, DependencyStatus, HealthResponse
from sunwai.server.services.process import process_cmdline

log = logging.getLogger(__name__)

DEFAULT_PORTS: dict[str, list[int]] = {
    "asr": [18931],
    "mt": [18933, 18981, 18982, 18983, 18984],
    "it2": [18936],
    "tts": list(range(18990, 19000)),
    "sep": [18952],
}
LIP_MODES = ("proc", "none")
WORKER_ID_RE = re.compile(r"[A-Za-z0-9_.\-]+")
CACHE_TTL_S = 3.0


@dataclass
class ServiceMap:
    """Where the pipeline's dependencies live.

    Attributes:
        deps: ``name -> ["host:port", ...]``.
        lip_mode: ``proc`` (lip-sync workers run as local processes) or ``none``.
        lip_workers: Identifiers of the lip-sync workers.
    """

    deps: dict[str, list[str]] = field(default_factory=dict)
    lip_mode: str = "proc"
    lip_workers: list[str] = field(default_factory=lambda: ["wA", "wB"])


def valid_worker_ids(workers: object) -> TypeGuard[list[str]]:
    """True for a non-empty list of unique identifiers made of ``[A-Za-z0-9_.-]``."""
    return (
        isinstance(workers, list)
        and bool(workers)
        and all(isinstance(w, str) and WORKER_ID_RE.fullmatch(w) for w in workers)
        and len(set(workers)) == len(workers)
    )


def _apply_services_file(path: str, services: ServiceMap) -> tuple[str | None, object]:
    """Merge a ``DUB_SERVICES`` JSON file into ``services``; return its ``lip_mode`` and ``lip_workers`` values."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for name, endpoints in data.items():
        if not isinstance(endpoints, list) or name == "lip_workers":
            continue
        if endpoints and all(":" in str(e) for e in endpoints):
            services.deps[name] = [str(e) for e in endpoints]
        else:
            log.warning("[services] ignoring empty/invalid list for %s", name)
    return data.get("lip_mode"), data.get("lip_workers")


def _select_lip_settings(
    services: ServiceMap, file_mode: str | None, file_workers: object, environ: Mapping[str, str]
) -> None:
    mode = file_mode
    if mode is not None and mode not in LIP_MODES:
        log.warning("[services] invalid lip_mode %r, using env/default", mode)
        mode = None
    if mode is None:
        env_mode = environ.get("LIP_MODE", "proc")
        mode = env_mode if env_mode in LIP_MODES else "proc"
    services.lip_mode = mode
    if file_workers is not None and valid_worker_ids(file_workers):
        services.lip_workers = list(file_workers)
        return
    if file_workers is not None:
        log.warning("[services] invalid lip_workers %r (need unique non-empty ids), using env/default", file_workers)
    env_workers = [w.strip() for w in environ.get("LIP_WORKERS", "wA,wB").split(",") if w.strip()]
    if not valid_worker_ids(env_workers):
        log.warning("[services] invalid env LIP_WORKERS %r (need unique ids), using wA,wB", env_workers)
        env_workers = ["wA", "wB"]
    services.lip_workers = env_workers


def load_services(environ: Mapping[str, str]) -> ServiceMap:
    """Resolve dependency endpoints; never hard-codes a node.

    Source: the JSON file named by ``DUB_SERVICES`` (optionally with ``lip_mode`` and ``lip_workers``), else the
    built-in port lists on host ``DEP_HOST`` (default 127.0.0.1).

    Args:
        environ: Environment-style mapping.

    Returns:
        The resolved :class:`ServiceMap`.
    """
    host = environ.get("DEP_HOST", "127.0.0.1")
    services = ServiceMap(deps={k: [f"{host}:{p}" for p in v] for k, v in DEFAULT_PORTS.items()})
    file_mode: str | None = None
    file_workers: object = None
    path = environ.get("DUB_SERVICES", "")
    if path:
        try:
            file_mode, file_workers = _apply_services_file(path, services)
        except Exception as exc:
            log.warning("[services] cannot read %s: %r", path, exc)
    _select_lip_settings(services, file_mode, file_workers, environ)
    return services


async def probe(endpoint: str, path: str = "/health", timeout_s: float = 1.5) -> bool:
    """Return True when ``host:port`` answers ``GET <path>`` with HTTP 200 within ``timeout_s``."""
    host, _, port = endpoint.rpartition(":")
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout_s)
        try:
            writer.write(f"GET {path} HTTP/1.0\r\nHost: x\r\n\r\n".encode())
            await writer.drain()
            head = await asyncio.wait_for(reader.read(64), timeout_s)
        finally:
            writer.close()
        return head.startswith(b"HTTP/1.") and b" 200" in head.split(b"\r\n", 1)[0]
    except Exception:
        return False


class HealthService:
    """Reports process health and probes the dependency services (results cached for a few seconds)."""

    def __init__(self, config: ServerConfig, services: ServiceMap, admission: JobAdmission) -> None:
        """Create the service.

        Args:
            config: Server configuration (reported verbatim under ``admission``).
            services: Dependency endpoints to probe.
            admission: Admission controller whose state is reported.
        """
        self._config = config
        self._services = services
        self._admission = admission
        self._cached_at = 0.0
        self._cached: tuple[dict[str, DependencyStatus], dict[str, int | None]] | None = None

    def lip_worker_pids(self) -> dict[str, int | None]:
        """Map each lip-sync worker id to its local pid, ``None`` when not running."""
        found: dict[str, int | None] = {w: None for w in self._services.lip_workers}
        if self._services.lip_mode != "proc":
            return found
        try:
            pids = [int(p) for p in os.listdir("/proc") if p.isdigit()]
        except OSError:
            return found
        for pid in pids:
            cmdline = process_cmdline(pid)
            if "sunwai.workers.lipsync" not in cmdline:
                continue
            for worker in self._services.lip_workers:
                if f"--id {worker} " in cmdline + " ":
                    found[worker] = pid
        return found

    async def dependency_status(self) -> tuple[dict[str, DependencyStatus], dict[str, int | None]]:
        """Probe every dependency endpoint and look up the lip-sync workers (cached for 3 s)."""
        if self._cached is not None and time.time() - self._cached_at < CACHE_TTL_S:
            return self._cached
        targets = [(name, ep) for name, eps in self._services.deps.items() for ep in eps]
        results = await asyncio.gather(*[probe(ep) for _, ep in targets])
        deps = {n: DependencyStatus(up=0, total=len(eps), down=[]) for n, eps in self._services.deps.items()}
        for (name, endpoint), ok in zip(targets, results, strict=True):
            if ok:
                deps[name].up += 1
            else:
                deps[name].down.append(endpoint)
        lip = await asyncio.get_running_loop().run_in_executor(None, self.lip_worker_pids)
        self._cached, self._cached_at = (deps, lip), time.time()
        return self._cached

    async def down_hint(self) -> str:
        """Return ``' (services unavailable: tts 9/10, ...)'`` for error messages, or ``''``."""
        try:
            deps, _ = await self.dependency_status()
        except Exception:
            return ""
        bits = [f"{n} {d.up}/{d.total}" for n, d in deps.items() if d.up < d.total]
        return f" (services unavailable: {', '.join(bits)})" if bits else ""

    async def report(self) -> HealthResponse:
        """Assemble the ``/health`` payload."""
        deps, lip = await self.dependency_status()
        degraded = [f"{n}:{d.up}/{d.total} down={d.down}" for n, d in deps.items() if d.up < d.total]
        if self._services.lip_mode == "proc":
            degraded += [f"lip:{worker} down" for worker, pid in lip.items() if pid is None]
        cfg = self._config
        admission = AdmissionInfo(
            on=cfg.admission,
            max_jobs=cfg.max_uncached_jobs,
            running=list(self._admission.running),
            queued=list(self._admission.queue),
            targets_max=cfg.max_targets,
            max_queued=cfg.max_queued_jobs,
            max_dur_s=cfg.max_dur_s,
            max_work_s=cfg.max_work_s,
            fanout=cfg.fanout,
            token_required=bool(cfg.token),
        )
        return HealthResponse(
            ok=True,
            ready=not degraded,
            degraded=degraded,
            root=str(cfg.root),
            pipeline=True,
            max_concurrent=cfg.max_concurrent,
            admission=admission,
            deps=deps,
            lip=lip,
            pid=os.getpid(),
            time=utcnow(),
        )
