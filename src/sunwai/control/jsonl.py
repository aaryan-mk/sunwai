"""Append-only JSON-lines files with fsync on write and crash-tolerant reads."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from sunwai.control.codec import Record, canon

TAIL_WINDOW_BYTES = 1 << 20


def parse_lines(lines: list[str] | list[bytes]) -> list[Record]:
    """Parse JSON lines, silently skipping any that are torn or invalid."""
    out: list[Record] = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


class JsonlFile:
    """One append-only ``.jsonl`` log: every append is flushed and fsynced, torn lines are ignored on read."""

    def __init__(self, path: Path) -> None:
        """Bind the log to ``path`` (created lazily on first append)."""
        self.path = path

    def append(self, record: Any) -> None:
        """Append one record as a canonical JSON line and fsync it."""
        line = canon(record) + "\n"
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def read_all(self) -> list[Record]:
        """Return every parseable record in file order."""
        if not self.path.exists():
            return []
        return parse_lines(self.path.read_text(encoding="utf-8").splitlines())

    def read_tail(self, count: int) -> list[Record]:
        """Return the last ``count`` parseable records without reading the whole file."""
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        with open(self.path, "rb") as handle:
            back = min(size, TAIL_WINDOW_BYTES)
            handle.seek(size - back)
            data = handle.read()
        lines = data.splitlines()
        if back < size:
            lines = lines[1:]  # the first line may be cut by the seek
        return parse_lines(lines[-count:])

    def rotate_keeping(self, records: list[Record], max_bytes: int) -> None:
        """Move an oversized log to ``<name>.1`` and restart it with ``records``; best effort."""
        try:
            if self.path.exists() and self.path.stat().st_size > max_bytes:
                os.replace(self.path, self.path.with_name(self.path.name + ".1"))
                with open(self.path, "w", encoding="utf-8") as handle:
                    for record in records:
                        handle.write(canon(record) + "\n")
        except OSError:
            pass
