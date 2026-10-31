"""Launching, bounding and cleaning up the planner and pipeline child processes."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import BinaryIO

from sunwai.server.atomic import atomic_write_json
from sunwai.server.clock import utcnow

log = logging.getLogger(__name__)

TAIL_BYTES = 6000


def kill_process_group(pgid: int, grace_s: float = 5.0) -> None:
    """SIGTERM a process group, then SIGKILL whatever is left after ``grace_s`` seconds (blocking, short).

    Args:
        pgid: Process group id.
        grace_s: Seconds to wait between SIGTERM and SIGKILL.
    """
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    started = time.time()
    while time.time() - started < grace_s:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)


def process_cmdline(pid: int) -> str:
    """Return a process's command line (NUL-separated arguments joined by spaces), ``""`` if unreadable."""
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def tail_since(path: Path, offset: int, nbytes: int = 4000) -> str:
    """Return the end of a log file, never reaching back before ``offset``.

    Args:
        path: Log file.
        offset: Position where the current run started.
        nbytes: Maximum number of trailing bytes.

    Returns:
        Decoded text, ``""`` if the file cannot be read.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(offset, size - nbytes))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


class ProcessRunner:
    """Runs ``python -m <module>`` children: never through a shell, in their own session, with a total timeout.

    Output goes to a log file (no unbounded in-memory buffer). Every child is registered under ``procs_dir`` so a
    restarted server can kill the orphans of a dead predecessor.
    """

    def __init__(self, job_root: Path, procs_dir: Path, package_src: Path, orphan_markers: tuple[str, ...]) -> None:
        """Create the runner.

        Args:
            job_root: Job root; only processes whose command line contains it count as orphans.
            procs_dir: Registry directory for live children.
            package_src: Directory prepended to the children's ``PYTHONPATH``.
            orphan_markers: Module names that identify our children when sweeping.
        """
        self._root = job_root
        self._procs_dir = procs_dir
        self._package_src = package_src
        self._markers = orphan_markers

    async def run(
        self,
        module: str,
        args: list[str],
        cwd: Path,
        log_path: Path,
        extra_env: dict[str, str] | None = None,
        timeout: float = 3600,
        kind: str = "job",
    ) -> tuple[int, str]:
        """Run a module to completion.

        Args:
            module: Dotted module name executed with ``-m``.
            args: Command-line arguments.
            cwd: Working directory.
            log_path: File that receives stdout and stderr (appended).
            extra_env: Variables added to the inherited environment.
            timeout: Total wall-time limit in seconds; the process group is killed past it.
            kind: Label for logs and the registry (``chunk``, ``planner``, ``resynth``).

        Returns:
            ``(exit code, tail of the log)``. 124 means timeout, 127 a missing interpreter.
        """
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            logf = open(log_path, "ab")
        except OSError as exc:
            return 1, f"cannot open log: {exc!r}"
        offset = logf.seek(0, 2)
        header = f"\n[server {utcnow()}] start {kind}: {module} {' '.join(args)[:300]}\n"
        logf.write(header.encode("utf-8", "replace"))
        logf.flush()
        spawned = await self._spawn(module, args, cwd, logf, extra_env)
        if isinstance(spawned, tuple):
            logf.close()
            return spawned
        registry = self._register(spawned.pid, module, kind, cwd)
        rc, note = await self._wait(spawned, timeout, kind, registry, logf)
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(spawned.pid, signal.SIGKILL)  # reap stragglers (ffmpeg grandchildren holding GPU/files)
        return rc, tail_since(log_path, offset, TAIL_BYTES) + note

    def _environment(self, extra_env: dict[str, str] | None) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(self._package_src), env.get("PYTHONPATH", "")]))
        env.update(extra_env or {})
        return env

    async def _spawn(
        self, module: str, args: list[str], cwd: Path, logf: BinaryIO, extra_env: dict[str, str] | None
    ) -> asyncio.subprocess.Process | tuple[int, str]:
        try:
            return await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                module,
                *args,
                cwd=str(cwd),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=logf,
                stderr=asyncio.subprocess.STDOUT,
                env=self._environment(extra_env),
                start_new_session=True,
            )
        except FileNotFoundError:
            return 127, "Python executable not found"
        except Exception as exc:
            return 1, f"spawn error: {exc!r}"

    def _register(self, pid: int, module: str, kind: str, cwd: Path) -> Path:
        entry = self._procs_dir / f"{pid}.json"
        record = {
            "pid": pid,
            "pgid": pid,
            "kind": kind,
            "script": module,
            "cwd": str(cwd),
            "started": utcnow(),
            "server_pid": os.getpid(),
        }
        with contextlib.suppress(OSError):
            self._procs_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(entry, record)
        return entry

    async def _wait(
        self, proc: asyncio.subprocess.Process, timeout: float, kind: str, registry: Path, logf: BinaryIO
    ) -> tuple[int, str]:
        loop = asyncio.get_running_loop()
        note = ""
        rc: int | None = None
        try:
            try:
                rc = await asyncio.wait_for(proc.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                note = f"\n[server] {kind} timed out after {timeout:.0f}s; process group killed\n"
                await loop.run_in_executor(None, kill_process_group, proc.pid, 5.0)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(proc.wait(), timeout=10)
                rc = 124
        except BaseException:
            # server task cancelled / shutting down: never leave the child group behind
            await loop.run_in_executor(None, kill_process_group, proc.pid, 2.0)
            raise
        finally:
            with contextlib.suppress(OSError):
                registry.unlink()
            self._close_log(logf, note)
        return (rc if rc is not None else 1), note

    @staticmethod
    def _close_log(logf: BinaryIO, note: str) -> None:
        if note:
            with contextlib.suppress(OSError):
                logf.write(note.encode("utf-8"))
                logf.flush()
        logf.close()

    def sweep_orphans(self) -> int:
        """Kill pipeline/planner process groups under the job root left behind by a dead predecessor.

        At startup nothing is our child yet, so any such process would race the retry.

        Returns:
            Number of process groups killed.
        """
        killed: set[int] = set()
        me = os.getpid()
        try:
            pids = [int(p) for p in os.listdir("/proc") if p.isdigit()]
        except OSError:
            pids = []
        for pid in pids:
            if pid == me:
                continue
            cmdline = process_cmdline(pid)
            if not (cmdline and any(m in cmdline for m in self._markers) and str(self._root) in cmdline):
                continue
            try:
                pgid = os.getpgid(pid)
            except OSError:
                continue
            if pgid not in killed and pgid != os.getpgid(0):
                log.warning("[sweep] killing orphan pid=%s pgid=%s: %s", pid, pgid, cmdline[:160])
                kill_process_group(pgid)
                killed.add(pgid)
        with contextlib.suppress(OSError):
            for entry in self._procs_dir.glob("*.json"):
                entry.unlink()
        return len(killed)
