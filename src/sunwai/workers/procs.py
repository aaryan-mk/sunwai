"""Stdlib-only process and HTTP helpers for the supervisor."""

from __future__ import annotations

import os
import signal
import time
import urllib.request

EXCLUDED_CMDLINE_MARKER = "supervisor_app2"


def atomic_write(path: str, text: str) -> None:
    """Write ``text`` to ``path`` durably and atomically (temp file, fsync, rename)."""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def http_ok(url: str, timeout: float = 3.0) -> bool:
    """True when a GET of ``url`` answers 200 within ``timeout`` seconds."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return bool(response.status == 200)
    except Exception:
        return False


def read_cmdline(pid: int) -> str:
    """Command line of a process with arguments separated by spaces; empty when it is gone."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def pids_matching(pattern: str) -> list[int]:
    """Pids (other than ours) whose command line contains ``pattern``."""
    me = os.getpid()
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == me:
            continue
        cmdline = read_cmdline(int(entry))
        if cmdline and pattern in cmdline and EXCLUDED_CMDLINE_MARKER not in cmdline:
            pids.append(int(entry))
    return pids


def kill_all(pids: list[int], grace: float = 8) -> None:
    """SIGTERM the processes, wait up to ``grace`` seconds, then SIGKILL the survivors."""
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    started = time.time()
    while time.time() - started < grace and any(os.path.exists(f"/proc/{pid}") for pid in pids):
        time.sleep(0.3)
    for pid in pids:
        if os.path.exists(f"/proc/{pid}"):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
