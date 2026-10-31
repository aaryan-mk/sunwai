"""Job-level exclusive lock: planner retries and concurrent planners serialise on it."""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def job_lock(out_dir: str) -> Iterator[None]:
    """Hold an exclusive ``flock`` on ``<out_dir>/.planner.lock`` for the duration of the ``with`` block."""
    with open(os.path.join(out_dir, ".planner.lock"), "a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
