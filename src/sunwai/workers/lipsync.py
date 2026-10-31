"""Model-resident lip-sync worker for a job root (``DUB_ROOT``).

usage: python -m sunwai.workers.lipsync --root ./jobs --id w7 [--once] [--cfg '{"steps":20}']

The process changes into ``LATENTSYNC_DIR`` (the upstream LatentSync checkout) so its relative ``configs/`` and
``checkpoints/`` paths resolve.

* a chunk is claimed with an ``O_EXCL`` lock file (``lipsync.lock``, heartbeat-touched, stale after 90 s), so N
  workers are safe;
* priority: the chunk the viewer is watching (``<root>/focus.json`` if the server writes it, else the newest
  ``/media/<job>/<lang>/chunks/<n>.mp4`` request in ``server.log``), then the chunks after it, then everything
  else, newest dub first;
* output is only published if the dub was not rewritten (segment edit) while we worked.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Protocol

from sunwai.workers.candidates import Candidate, CandidateScanner, Focus, next_failure_count, rank, read_focus
from sunwai.workers.claims import ChunkLock, Heartbeat
from sunwai.workers.common import file_signature, log

IDLE_SLEEP_S = 3


class ChunkRenderer(Protocol):
    """What the worker needs from the lip-sync engine."""

    def process_chunk(
        self, job_dir: str, lang: str, out_path: str | None = None, report_path: str | None = None, **overrides: Any
    ) -> dict[str, Any]:
        """Render one chunk and return its report."""
        ...

    def release_cache(self) -> None:
        """Give cached GPU memory back."""
        ...


def _remove_quietly(path: str) -> None:
    """Delete a file, ignoring a missing file or any OS error."""
    try:
        os.remove(path)
    except OSError:
        pass


class LipSyncWorker:
    """Claims dubbed chunks and renders their lip-synced video.

    Args:
        root: Job root.
        worker_id: Name used in logs and in the stats file.
        engine: Resident lip-sync engine.
        options: Per-chunk engine options (``--cfg``).
    """

    def __init__(self, root: str, worker_id: str, engine: ChunkRenderer, options: dict[str, Any]) -> None:
        self.root = root
        self.worker_id = worker_id
        self.engine = engine
        self.options = options
        self.scanner = CandidateScanner(root)
        self.stats_path = os.path.join(root, f"lipsync_stats_{worker_id}.jsonl")

    def _log(self, message: str) -> None:
        log(f"[{self.worker_id}] {message}")

    def run(self, once: bool = False) -> None:
        """Process chunks until stopped (or until nothing is left, with ``once``)."""
        while True:
            took = self.run_pass()
            if once and not took:
                return
            if not took:
                time.sleep(IDLE_SLEEP_S)
            elif once:
                return

    def run_pass(self) -> bool:
        """Claim and process at most one chunk (best-ranked first).

        Returns:
            True when a chunk was claimed.
        """
        focus = read_focus(self.root)
        candidates = sorted(self.scanner.scan(), key=lambda c: rank(c, focus))
        claimed = False
        for candidate in candidates:
            lock = ChunkLock.acquire(candidate.dir)
            if lock is None:
                continue
            claimed = True
            if not os.path.isdir(candidate.dir) or not os.path.exists(
                candidate.dub
            ):  # vanished between listing and claim
                self._log(f"SKIP vanished {candidate.dir}")
                lock.release()
                break
            if self._work(candidate, lock, focus):
                break
        return claimed

    def _work(self, candidate: Candidate, lock: ChunkLock, focus: Focus | None) -> bool:
        """Render one claimed chunk; returns False when another worker already finished it (keep scanning)."""
        heartbeat = Heartbeat(lock)
        heartbeat.start()
        started = time.time()
        self._log(f"START {candidate.job}/{candidate.lang}/c{candidate.idx:04d} focus={focus}")
        try:
            lipsync = candidate.lipsync_path
            if os.path.exists(lipsync) and os.path.getmtime(lipsync) >= os.path.getmtime(candidate.dub):
                return False
            self._render_and_publish(candidate, lock, started)
        except Exception as ex:
            self._log(f"FAILED {candidate.dir}: {ex}")
            traceback.print_exc()
            self._record_failure(candidate, ex)
        finally:
            self._release(heartbeat, lock)
        return True

    def _render_and_publish(self, candidate: Candidate, lock: ChunkLock, started: float) -> None:
        """Render to side files and publish them unless the dub changed or the lock was lost meanwhile."""
        lipsync = candidate.lipsync_path
        dub_signature = file_signature(candidate.dub)
        new_video = lipsync + f".{os.getpid()}.new.mp4"
        report_path = os.path.join(candidate.dir, "lipsync.json")
        new_report = report_path + f".{os.getpid()}"
        report = self.engine.process_chunk(
            candidate.dir, candidate.lang, out_path=new_video, report_path=new_report, **self.options
        )
        # TOCTOU: re-check the dub signature (mtime_ns + inode + size) immediately before the rename and again
        # right after it; if the dub was rewritten (segment edit) in between, roll the published file back.
        stale = file_signature(candidate.dub) != dub_signature or not lock.owned()
        if not stale:
            mine = file_signature(new_video)
            os.replace(new_video, lipsync)
            if file_signature(candidate.dub) != dub_signature or not lock.owned():
                self._roll_back(lipsync, mine)
                stale = True
        if stale:
            for path in (new_video, new_report):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
            self._log(f"DISCARD stale/lost lock {candidate.dir}")
            return
        os.replace(new_report, report_path)
        wall = time.time() - started
        self._log(
            f"DONE {candidate.job}/{candidate.lang}/c{candidate.idx:04d} {wall:.0f}s "
            f"rtf={report['rtf']} runs={len(report['runs'])}"
        )
        self._append_stats(candidate, wall, report)

    @staticmethod
    def _roll_back(published: str, mine: tuple[int, int, int]) -> None:
        """Remove the published video only if it is still ours (same inode and size), never a newer worker's."""
        try:
            if file_signature(published)[1:] == mine[1:]:
                os.remove(published)
        except OSError:
            pass

    def _append_stats(self, candidate: Candidate, wall: float, report: dict[str, Any]) -> None:
        """Append one line to the worker's stats file."""
        entry = {
            "ts": time.time(),
            "dir": candidate.dir,
            "wall": round(wall, 1),
            "rtf": report["rtf"],
            "chunk_s": report["chunk_s"],
            "synced_frames": report["synced_frames"],
            "timing": report["timing_s"],
        }
        with open(self.stats_path, "a") as fh:
            fh.write(json.dumps(entry) + "\n")

    def _record_failure(self, candidate: Candidate, error: Exception) -> None:
        """Write the ``lipsync.failed`` marker (``<count>|<message>``) used for retry back-off."""
        marker = os.path.join(candidate.dir, "lipsync.failed")
        failures = next_failure_count(marker, candidate.dub)
        try:  # a vanished job dir must never kill the worker
            with open(marker, "w") as fh:
                fh.write(f"{failures}|{error}")
        except OSError as ex:
            self._log(f"cannot write failure marker (job gone?): {ex}")

    def _release(self, heartbeat: Heartbeat, lock: ChunkLock) -> None:
        """Free this worker's cached GPU memory (OOM guard) and drop the lock after every chunk, ok or failed."""
        try:
            self.engine.release_cache()
        except Exception as ex:
            self._log(f"release_cache error: {ex}")
        heartbeat.stop()
        lock.release()


def main() -> None:
    """Command line entry point: load the engine inside the LatentSync checkout and run the worker."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--id", default="w")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--cfg", default="{}")
    args = parser.parse_args()
    latentsync_dir = os.environ.get("LATENTSYNC_DIR", "").strip()
    if latentsync_dir:
        os.chdir(latentsync_dir)
    sys.path.insert(0, os.getcwd())
    from sunwai.lipsync.engine import Engine

    LipSyncWorker(args.root, args.id, Engine(), json.loads(args.cfg)).run(once=args.once)


if __name__ == "__main__":
    main()
