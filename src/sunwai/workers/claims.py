"""Interprocess chunk claims: an exclusive lock file with a heartbeat, so several workers can share a job root."""

from __future__ import annotations

import os
import socket
import threading
import time

STALE_AFTER_S = 90
HEARTBEAT_S = 15
LOCK_NAME = "lipsync.lock"


class ChunkLock:
    """An ownership token written into ``<chunk dir>/lipsync.lock``.

    Args:
        path: Lock file path.
        token: Unique token identifying the owner.
    """

    def __init__(self, path: str, token: str) -> None:
        self.path = path
        self.token = token

    @classmethod
    def acquire(cls, chunk_dir: str) -> ChunkLock | None:
        """Claim a chunk with an ``O_EXCL`` lock file, taking over a stale lock if there is one.

        Args:
            chunk_dir: Chunk directory.

        Returns:
            The lock, or None when another worker holds it or the directory vanished.
        """
        path = os.path.join(chunk_dir, LOCK_NAME)
        for _ in range(2):
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileNotFoundError:  # the chunk dir vanished (job deleted/restored mid-queue)
                return None
            except FileExistsError:
                if not cls._take_over_stale(path):
                    return None
                continue
            token = f"{socket.gethostname()}-{os.getpid()}-{time.time()}"
            os.write(fd, token.encode())
            os.close(fd)
            return cls(path, token)
        return None

    @staticmethod
    def _take_over_stale(path: str) -> bool:
        """Remove a lock nobody has touched for ``STALE_AFTER_S``; True when the claim should be retried."""
        try:
            if time.time() - os.path.getmtime(path) <= STALE_AFTER_S:
                return False
            gone = f"{path}.stale.{os.getpid()}.{time.time()}"
            os.rename(path, gone)  # atomic: only one taker wins the stale lock
            try:
                os.remove(gone)
            except OSError:
                pass
            return True
        except OSError:
            return False

    def owned(self) -> bool:
        """True while the lock file still holds our token."""
        try:
            with open(self.path) as fh:
                return fh.read() == self.token
        except OSError:
            return False

    def release(self) -> None:
        """Delete the lock file, but only if it is still ours."""
        if self.owned():
            try:
                os.remove(self.path)
            except OSError:
                pass


class Heartbeat(threading.Thread):
    """Touches a lock file every ``HEARTBEAT_S`` seconds while the owner works.

    Args:
        lock: The lock to keep fresh.
    """

    def __init__(self, lock: ChunkLock) -> None:
        super().__init__(daemon=True)
        self._lock = lock
        self._stopped = threading.Event()

    def run(self) -> None:
        """Touch the lock until stopped."""
        while not self._stopped.wait(HEARTBEAT_S):
            try:
                if self._lock.owned():
                    os.utime(self._lock.path, None)
            except OSError:
                pass

    def stop(self) -> None:
        """Stop touching the lock and wait briefly for the thread."""
        self._stopped.set()
        self.join(timeout=5)
