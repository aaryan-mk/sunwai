"""Finding and ranking the chunks that still need a lip-synced video."""

from __future__ import annotations

import glob
import json
import os
import re
import time
from dataclasses import dataclass

from sunwai.workers.common import log

FOCUS_MAX_AGE_S = 900
LOG_TAIL_BYTES = 300000
DUB_SETTLE_S = 3
MAX_FAILURES = 3
RETRY_BACKOFF_S = 60
CHUNK_RE = re.compile(r"/media/([0-9a-f]+)/([A-Za-z_]+)/chunks/(\d+)\.mp4")
CHUNK_DIR_RE = re.compile(r"c\d{4}")
REQUIRED_FILES = ("dub.wav", "segments.json", "src.mp4")


@dataclass(frozen=True)
class Focus:
    """The chunk a viewer is watching.

    Attributes:
        job: Job id.
        lang: Language code.
        chunk: Chunk index.
    """

    job: str
    lang: str
    chunk: int


@dataclass(frozen=True)
class Candidate:
    """A dubbed chunk that may need lip-syncing.

    Attributes:
        dir: Chunk directory.
        lang: Language code.
        job: Job id.
        idx: Chunk index (0 for single-chunk jobs).
        dub: Path of the dubbed video.
        dub_mtime: Modification time of the dubbed video.
    """

    dir: str
    lang: str
    job: str
    idx: int
    dub: str
    dub_mtime: float

    @property
    def lipsync_path(self) -> str:
        """Where the lip-synced video is published."""
        return os.path.join(self.dir, f"dub_{self.lang}_lipsync.mp4")


def read_focus(root: str) -> Focus | None:
    """Return what the viewer is watching: ``focus.json`` if fresh, else the newest chunk request in ``server.log``."""
    focus_file = os.path.join(root, "focus.json")
    try:
        if time.time() - os.path.getmtime(focus_file) < FOCUS_MAX_AGE_S:
            with open(focus_file) as fh:
                data = json.load(fh)
            return Focus(data["job"], data["lang"], int(data["chunk"]))
    except Exception:
        pass
    return _focus_from_server_log(os.path.join(root, "server.log"))


def _focus_from_server_log(path: str) -> Focus | None:
    """Newest ``/media/<job>/<lang>/chunks/<n>.mp4`` request in the tail of a fresh server log."""
    try:
        if time.time() - os.path.getmtime(path) > FOCUS_MAX_AGE_S:
            return None
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - LOG_TAIL_BYTES))
            text = fh.read().decode("utf-8", "ignore")
    except Exception:
        return None
    matches = list(CHUNK_RE.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    return Focus(last.group(1), last.group(2), int(last.group(3)))


def lipsync_disabled(root: str, job: str) -> bool:
    """True only when ``<root>/<job>/job.json`` has ``"lipsync": false`` (missing or unreadable is False)."""
    try:
        with open(os.path.join(root, job, "job.json")) as fh:
            return bool(json.load(fh).get("lipsync") is False)
    except Exception:
        return False


def parse_chunk_dir(chunk_dir: str) -> tuple[str, int]:
    """Split a chunk directory into ``(job, chunk index)``.

    Chunked jobs live in ``<job>/<lang>/c0007``; single-chunk jobs in ``<job>/<lang>`` (index 0).
    """
    parts = chunk_dir.split("/")
    if CHUNK_DIR_RE.fullmatch(parts[-1]):
        return parts[-3], int(parts[-1][1:])
    return parts[-2], 0


def retry_blocked(failures: int, seconds_since_failure: float) -> bool:
    """True when a chunk that failed ``failures`` times must not be retried yet (3 strikes, 60 s back-off each)."""
    return failures >= MAX_FAILURES or seconds_since_failure < RETRY_BACKOFF_S * failures


def read_failure_count(marker: str) -> int:
    """Failure count stored in a ``lipsync.failed`` marker (``<count>|<message>``); 99 when unreadable."""
    try:
        with open(marker) as fh:
            return int(fh.read().split("|")[0])
    except Exception:
        return 99


def next_failure_count(marker: str, dub: str) -> int:
    """Failure count to write next: stored count + 1 if the marker is newer than the dub, else 1."""
    try:
        if os.path.exists(marker) and os.path.getmtime(marker) >= os.path.getmtime(dub):
            with open(marker) as fh:
                return int(fh.read().split("|")[0]) + 1
    except Exception:
        pass
    return 1


def rank(candidate: Candidate, focus: Focus | None) -> tuple[int, int, int, int, float]:
    """Sort key: the watched job first (same language, then chunks ahead of the viewer by distance, then behind),
    everything else newest dub first.
    """
    if focus and candidate.job == focus.job:
        same_lang = 0 if candidate.lang == focus.lang else 1
        ahead = candidate.idx - focus.chunk
        return (0, same_lang, 1 if ahead < 0 else 0, abs(ahead), 0.0)
    return (1, 0, 0, 0, -candidate.dub_mtime)


def _needs_render(dub: str, lipsync: str, chunk_dir: str) -> float | None:
    """The dub's mtime when the chunk still needs rendering, else None (up to date, too fresh or backing off)."""
    try:
        dub_mtime = os.path.getmtime(dub)
        if time.time() - dub_mtime < DUB_SETTLE_S:
            return None
        if os.path.exists(lipsync) and os.path.getmtime(lipsync) >= dub_mtime:
            return None
        marker = os.path.join(chunk_dir, "lipsync.failed")
        if os.path.exists(marker) and os.path.getmtime(marker) >= dub_mtime:
            failures = read_failure_count(marker)
            if retry_blocked(failures, time.time() - os.path.getmtime(marker)):
                return None
    except OSError:
        return None
    return dub_mtime


class CandidateScanner:
    """Scans a job root for chunks that need lip-syncing.

    Args:
        root: Job root (``DUB_ROOT``).
    """

    def __init__(self, root: str) -> None:
        self.root = root
        self._skip_logged: set[str] = set()

    def scan(self) -> list[Candidate]:
        """List the chunks whose dub is newer than their lip-synced video and that the user did not opt out of."""
        found = []
        disabled: dict[str, bool] = {}  # job.json is read once per job per scan, so decisions are consistent
        dubs = glob.glob(f"{self.root}/*/*/c*/dub_*.mp4") + glob.glob(f"{self.root}/*/*/dub_*.mp4")
        for dub in dubs:
            if dub.endswith("_lipsync.mp4") or ".part" in dub or ".new" in dub:
                continue
            chunk_dir = os.path.dirname(dub)
            lang = os.path.basename(dub)[4:-4]
            dub_mtime = _needs_render(dub, os.path.join(chunk_dir, f"dub_{lang}_lipsync.mp4"), chunk_dir)
            if dub_mtime is None:
                continue
            if not all(os.path.exists(os.path.join(chunk_dir, name)) for name in REQUIRED_FILES):
                continue
            job, idx = parse_chunk_dir(chunk_dir)
            if job not in disabled:
                disabled[job] = lipsync_disabled(self.root, job)
            if disabled[job]:  # user turned lip-sync off for this video: no lock, no render, no files
                self._log_skip_once(job)
                continue
            found.append(Candidate(chunk_dir, lang, job, idx, dub, dub_mtime))
        return found

    def _log_skip_once(self, job: str) -> None:
        if job not in self._skip_logged:
            self._skip_logged.add(job)
            log(f"SKIP lipsync=false job {job}")
