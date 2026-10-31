"""Small helpers shared by the worker processes."""

from __future__ import annotations

import os
import time

FileSignature = tuple[int, int, int]


def log(*parts: object) -> None:
    """Print a timestamped line and flush, so supervisors tailing the log see it at once."""
    print(time.strftime("%F %T"), *parts, flush=True)


def file_signature(path: str) -> FileSignature:
    """Identity of a published file: ``(mtime_ns, inode, size)``.

    Atomic ``os.replace`` publication changes the inode, so a rewritten file never keeps its old signature.
    """
    stat = os.stat(path)
    return (stat.st_mtime_ns, stat.st_ino, stat.st_size)
