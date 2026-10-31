"""Rollback support for the write-ahead scheduling transaction."""

from __future__ import annotations

import os
import time
from pathlib import Path

from sunwai.jobs.scheduler import store


def merge_lang_into_job(job_dir: Path, lang: str) -> list[str]:
    """Add ``lang`` to ``job.json`` atomically and create its output directory.

    Returns:
        The job's language list after the merge.
    """
    job_dir = Path(job_dir)
    meta = store.read_json(job_dir / "job.json")
    langs = list(meta.get("langs") or [])
    if lang not in langs:
        langs.append(lang)
        meta["langs"] = langs
        meta["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
        store.atomic_json(job_dir / "job.json", meta)
    (job_dir / lang).mkdir(parents=True, exist_ok=True)
    return langs


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + f".rb{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class JobSnapshot:
    """Byte-level snapshot of ``job.json``, ``tracks.json`` and the language directory taken before a schedule."""

    def __init__(self, job_dir: Path, lang: str) -> None:
        """Capture the current state of ``job_dir`` for ``lang``."""
        self.job_dir = job_dir
        self.lang = lang
        self._job_path = job_dir / "job.json"
        self._tracks_path = job_dir / "tracks.json"
        self._job_bytes = self._job_path.read_bytes()
        self._tracks_bytes = self._tracks_path.read_bytes() if self._tracks_path.exists() else None
        self._lang_dir_existed = (job_dir / lang).exists()

    def restore(self, restore_job: bool) -> None:
        """Undo a failed schedule.

        Args:
            restore_job: True when the language was newly merged, so ``job.json`` and the language directory are
                rolled back too. ``tracks.json`` is always restored. Failure to restore is swallowed: the durable
                ``scheduling`` intent stays on disk and recovery drops it because the language is not in ``job.json``.
        """
        try:
            if restore_job:
                _write_bytes_atomic(self._job_path, self._job_bytes)
                if not self._lang_dir_existed:
                    try:
                        (self.job_dir / self.lang).rmdir()
                    except OSError:
                        pass
            if self._tracks_bytes is None:
                self._tracks_path.unlink(missing_ok=True)
            else:
                _write_bytes_atomic(self._tracks_path, self._tracks_bytes)
        except OSError:
            pass
