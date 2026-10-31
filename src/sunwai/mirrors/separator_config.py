"""Configuration of the separation mirror (kept free of torch/demucs so it can be tested)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SeparatorConfig:
    """Limits and root of the separation service.

    Attributes:
        root: Only input and output paths under this directory are accepted.
        max_pending: Queued requests before new ones get 503.
        queue_timeout: Seconds a request may wait for the model.
        max_bytes: Largest input wav (a 100 s stereo 44.1 kHz chunk is ~17 MB).
        decode_parallel: Concurrent audio decodes (memory bound).
    """

    root: str
    max_pending: int = 24
    queue_timeout: float = 900.0
    max_bytes: int = 256 * 1024 * 1024
    decode_parallel: int = 2

    @classmethod
    def from_env(cls) -> SeparatorConfig:
        """Read the ``SEP_*`` environment variables."""
        return cls(
            root=os.path.realpath(os.environ.get("SEP_ROOT", os.getcwd())),
            max_pending=int(os.environ.get("SEP_MAX_PENDING", "24")),
            queue_timeout=float(os.environ.get("SEP_QUEUE_TIMEOUT", "900")),
            max_bytes=int(os.environ.get("SEP_MAX_BYTES", str(256 * 1024 * 1024))),
            decode_parallel=int(os.environ.get("SEP_DECODE_PARALLEL", "2")),
        )

    def contains(self, path: str) -> bool:
        """Return True when ``path`` is the root or below it (component-wise: ``/amk-evil`` is not ``/amk``)."""
        real = os.path.realpath(path)
        return real == self.root or real.startswith(self.root.rstrip(os.sep) + os.sep)
