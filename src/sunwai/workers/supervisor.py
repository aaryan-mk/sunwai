"""SunwAI supervisor (stdlib only). Runs in the foreground of one process or scheduler step.

* keeps uvicorn (``sunwai.server.app:create_app``) alive: restarts only uvicorn, never the tunnel, so the public URL
  does not rotate;
* keeps the quick tunnel alive: adopts an existing cloudflared, restarts it only after the PUBLIC url failed 3x in a
  row, and then writes the new URL to ``tunnel_url.txt`` (+ ``tunnel_url_history.txt``);
* probes every dependency each cycle (ASR, MT x5, IT2, TTS x10, sep, both lip workers) and writes
  ``<ROOT>/supervisor_status.json`` atomically; logs every up/down transition. ``--once`` prints the status and exits.

env: SUP_ROOT SUP_PORT SUP_NO_TUNNEL=1 SUP_APPDIR SUP_CLOUDFLARED
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

from sunwai.workers.common import log
from sunwai.workers.procs import atomic_write, http_ok, kill_all, pids_matching, read_cmdline
from sunwai.workers.services import ServiceMap, load_services

CYCLE_S = 5
TUNNEL_CHECK_S = 30
FAILS_BEFORE_RESTART = 3
UVICORN_GRACE_S = 60
TUNNEL_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


@dataclass(frozen=True)
class SupervisorConfig:
    """Supervisor settings.

    Attributes:
        root: Job root holding logs and status files.
        port: Port uvicorn listens on.
        appdir: Working directory of uvicorn.
        cloudflared: cloudflared executable.
        no_tunnel: Do not manage a tunnel.
    """

    root: str
    port: int
    appdir: str
    cloudflared: str
    no_tunnel: bool

    @classmethod
    def from_env(cls) -> SupervisorConfig:
        """Read the ``SUP_*`` variables (``SUP_ROOT`` falls back to ``DUB_ROOT``)."""
        return cls(
            root=os.environ.get("SUP_ROOT", os.environ.get("DUB_ROOT", "./jobs")),
            port=int(os.environ.get("SUP_PORT", "18941")),
            appdir=os.environ.get("SUP_APPDIR", "."),
            cloudflared=os.environ.get("SUP_CLOUDFLARED", "cloudflared"),
            no_tunnel=os.environ.get("SUP_NO_TUNNEL") == "1",
        )

    @property
    def uvicorn_pattern(self) -> str:
        """Command-line fragment identifying our uvicorn."""
        return f"uvicorn --factory sunwai.server.app:create_app --host 127.0.0.1 --port {self.port}"

    @property
    def tunnel_pattern(self) -> str:
        """Command-line fragment identifying our tunnel."""
        return f"cloudflared tunnel --no-autoupdate --url http://127.0.0.1:{self.port}"


class Supervisor:
    """Keeps the app and its tunnel alive and reports dependency health.

    Args:
        config: Supervisor settings.
        services: Dependencies to probe.
    """

    def __init__(self, config: SupervisorConfig, services: ServiceMap) -> None:
        self.config = config
        self.services = services
        self._uvicorn_failures = 0
        self._tunnel_failures = 0
        self._uvicorn_grace_until = 0.0
        self._last_tunnel_check = 0.0
        self._previous_down: set[str] = set()

    def read_url(self) -> str:
        """Current public tunnel URL (empty when unknown)."""
        try:
            with open(f"{self.config.root}/tunnel_url.txt") as fh:
                return fh.read().strip()
        except OSError:
            return ""

    def probe_deps(self) -> tuple[dict[str, dict[str, Any]], dict[str, int | None]]:
        """Probe every service ``/health`` and find the lip-sync worker processes.

        Returns:
            ``(deps, lip)``: per-service up/total/down endpoints, and worker id to pid (None = down, -1 = not
            checkable from this host).
        """
        deps = {}
        for name, endpoints in self.services.deps.items():
            down = [e for e in endpoints if not http_ok(f"http://{e}/health", 1.5)]
            deps[name] = {"up": len(endpoints) - len(down), "total": len(endpoints), "down": down}
        if self.services.lip_mode != "proc":
            return deps, {w: -1 for w in self.services.lip_workers}
        pids = pids_matching("sunwai.workers.lipsync")
        lip: dict[str, int | None] = {}
        for worker in self.services.lip_workers:
            matching = [p for p in pids if f"--id {worker} " in read_cmdline(p) + " "]
            lip[worker] = matching[0] if matching else None
        return deps, lip

    def spawn_uvicorn(self) -> None:
        """Start uvicorn detached, logging to ``server.log``."""
        cmd = [sys.executable, "-m", "uvicorn", "--factory", "sunwai.server.app:create_app", "--host", "127.0.0.1"]
        with open(f"{self.config.root}/server.log", "ab") as logfile:
            proc = subprocess.Popen(
                [*cmd, "--port", str(self.config.port)],
                cwd=self.config.appdir,
                env=dict(os.environ, DUB_ROOT=self.config.root),
                stdout=logfile,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        log(f"uvicorn spawned pid={proc.pid}")

    def spawn_tunnel(self) -> str:
        """Start a quick tunnel, wait for its URL and record it; returns the URL ('' if none appeared)."""
        tunnel_log = f"{self.config.root}/tunnel.log"
        open(tunnel_log, "w").close()
        with open(tunnel_log, "ab") as logfile:
            proc = subprocess.Popen(
                [self.config.cloudflared, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{self.config.port}"],
                stdout=logfile,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        url = ""
        for _ in range(40):
            time.sleep(2)
            with open(tunnel_log, errors="replace") as fh:
                match = TUNNEL_URL_RE.search(fh.read())
            if match:
                url = match.group(0)
                break
        log(f"tunnel spawned pid={proc.pid} url={url}")
        if url:
            atomic_write(f"{self.config.root}/tunnel_url.txt", url + "\n")
            with open(f"{self.config.root}/tunnel_url_history.txt", "a") as fh:
                fh.write(f"{time.strftime('%F %T')} {url}\n")
        return url

    def _check_uvicorn(self, now: float) -> tuple[bool, list[int]]:
        """Restart uvicorn when missing or unhealthy 3x in a row; returns ``(healthy, pids)``."""
        pids = pids_matching(self.config.uvicorn_pattern)
        healthy = http_ok(f"http://127.0.0.1:{self.config.port}/health", 4.0)
        if healthy:
            self._uvicorn_failures = 0
        elif not pids:
            log("uvicorn not running -> start")
            self.spawn_uvicorn()
            self._uvicorn_grace_until = now + UVICORN_GRACE_S
            self._uvicorn_failures = 0
        elif now > self._uvicorn_grace_until:
            self._uvicorn_failures += 1
            if self._uvicorn_failures >= FAILS_BEFORE_RESTART:
                log(f"uvicorn unhealthy x{self._uvicorn_failures} -> restart (pids {pids})")
                kill_all(pids)
                self.spawn_uvicorn()
                self._uvicorn_grace_until = time.time() + UVICORN_GRACE_S
                self._uvicorn_failures = 0
        return healthy, pids

    def _check_tunnel(self, now: float) -> None:
        """Check the public URL every 30 s; restart the tunnel (rotating the URL) after 3 failures."""
        if self.config.no_tunnel or now - self._last_tunnel_check < TUNNEL_CHECK_S:
            return
        self._last_tunnel_check = now
        url = self.read_url()
        if url and http_ok(url + "/health", 10.0):
            self._tunnel_failures = 0
        elif not url and not pids_matching(self.config.tunnel_pattern):
            log("no tunnel -> start")
            self.spawn_tunnel()
            self._tunnel_failures = 0
        else:
            self._tunnel_failures += 1
            log(f"public url check failed x{self._tunnel_failures} ({url})")
            if self._tunnel_failures >= FAILS_BEFORE_RESTART:
                log("restarting tunnel (URL will rotate)")
                kill_all(pids_matching(self.config.tunnel_pattern))
                new_url = self.spawn_tunnel()
                self._tunnel_failures = 0
                log(f"NEW PUBLIC URL {new_url}")

    def run_cycle(self) -> None:
        """One supervision cycle: app, tunnel, dependencies, status file."""
        now = time.time()
        healthy, uvicorn_pids = self._check_uvicorn(now)
        self._check_tunnel(now)
        deps, lip = self.probe_deps()
        down = {f"{n}:{e}" for n, info in deps.items() for e in info["down"]} | {
            f"lip:{w}" for w, pid in lip.items() if pid is None
        }
        for item in sorted(down - self._previous_down):
            log("DEP DOWN", item)
        for item in sorted(self._previous_down - down):
            log("DEP UP", item)
        self._previous_down = down
        status = {
            "time": time.strftime("%F %T"),
            "ready": healthy and not down,
            "app_healthy": healthy,
            "uvicorn_pids": uvicorn_pids,
            "tunnel_pids": pids_matching(self.config.tunnel_pattern),
            "url": self.read_url(),
            "tunnel_fail": self._tunnel_failures,
            "deps": deps,
            "lip": lip,
            "down": sorted(down),
        }
        try:
            atomic_write(f"{self.config.root}/supervisor_status.json", json.dumps(status, indent=1))
        except OSError:
            pass

    def run(self) -> None:
        """Supervise forever."""
        cfg = self.config
        log(
            f"supervisor start pid={os.getpid()} root={cfg.root} port={cfg.port} "
            f"tunnel={'off' if cfg.no_tunnel else 'on'}"
        )
        while True:
            started = time.time()
            self.run_cycle()
            time.sleep(max(0.5, CYCLE_S - (time.time() - started)))

    def print_status(self) -> None:
        """Print the current dependency and process status as JSON."""
        deps, lip = self.probe_deps()
        status = {
            "deps": deps,
            "lip": lip,
            "uvicorn": pids_matching(self.config.uvicorn_pattern),
            "tunnel": pids_matching(self.config.tunnel_pattern),
            "url": self.read_url(),
        }
        print(json.dumps(status, indent=1))


def main() -> None:
    """Command line entry point (``--once`` prints the status and exits)."""
    config = SupervisorConfig.from_env()
    os.makedirs(config.root, exist_ok=True)
    supervisor = Supervisor(config, load_services())
    if "--once" in sys.argv:
        supervisor.print_status()
    else:
        supervisor.run()


if __name__ == "__main__":
    main()
