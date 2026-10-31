"""Transcript segments: listing, and the transactional edit/undo flow that re-synthesises a line."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sunwai.server.atomic import JsonObject
from sunwai.server.config import ServerConfig
from sunwai.server.errors import ServiceError
from sunwai.server.paths import LANG_RE, SEG_RE, JobPaths
from sunwai.server.repository import JobRepository, Segments
from sunwai.server.scheduling import GateBusy, KeyedLocks, PriorityGate
from sunwai.server.schemas import EditResponse, SegmentList, SegmentView
from sunwai.server.services.chunks import PIPELINE_MODULE
from sunwai.server.services.process import ProcessRunner

Segment = dict[str, Any]
Change = Callable[[Segment], str]
After = Callable[[Segment], None]


def find_segment(segments: Segments, seg_id: str) -> tuple[int, Segment | None]:
    """Return ``(index, segment)`` of the segment with id ``seg_id``, ``(-1, None)`` if absent."""
    for index, segment in enumerate(segments):
        if str(segment.get("id")) == seg_id:
            return index, segment
    return -1, None


def apply_edit(segment: Segment, text: str) -> str:
    """Replace a segment's target text, pushing the old text onto its history; return the new text."""
    history = segment.get("history")
    if not isinstance(history, list):
        history = []
    history.append(segment.get("tgt_text", ""))
    segment["tgt_text"] = text
    segment["history"] = history
    segment["edited"] = True
    return text


def apply_undo(segment: Segment) -> tuple[str, list[Any]]:
    """Pop the last history entry back into the segment; return ``(restored text, remaining history)``.

    Raises:
        ServiceError: 409 when there is nothing to undo.
    """
    history = segment.get("history")
    if not isinstance(history, list) or not history:
        raise ServiceError(409, "nothing to undo")
    item = history.pop()
    # history entries are plain strings (this server) or dicts (older servers)
    text = (item.get("tgt_text") or item.get("text") or "") if isinstance(item, dict) else str(item)
    segment["tgt_text"] = text
    segment["history"] = history
    segment["edited"] = bool(history)
    return text, history


@dataclass(frozen=True)
class EditTarget:
    """Everything an edit needs, resolved and validated.

    Attributes:
        job_dir: Job directory.
        meta: Contents of ``job.json``.
        chunk: The planned chunk.
        out_dir: Chunk output directory.
        lang: Target language.
        seg_id: Segment identifier.
    """

    job_dir: Path
    meta: JsonObject
    chunk: JsonObject
    out_dir: Path
    lang: str
    seg_id: str


class SegmentService:
    """Reads transcript lines and applies edits transactionally (snapshot, re-synthesise, roll back on failure)."""

    def __init__(
        self,
        config: ServerConfig,
        repo: JobRepository,
        paths: JobPaths,
        locks: KeyedLocks,
        gate: PriorityGate,
        process: ProcessRunner,
    ) -> None:
        """Create the service.

        Args:
            config: Server configuration (edit limits, timeouts).
            repo: Job storage.
            paths: Directory resolver.
            locks: Keyed locks (one per chunk).
            gate: Pipeline slot gate (edits take it with top priority).
            process: Subprocess runner for the re-synthesis.
        """
        self._config = config
        self._repo = repo
        self._paths = paths
        self._locks = locks
        self._gate = gate
        self._process = process

    # ------------------------------------------------------------------ listing
    def list_segments(self, job_id: str, lang: str) -> SegmentList:
        """Return every line of finished chunks with global times, sorted by start.

        Raises:
            ServiceError: 404 for an unknown job or a language that is not part of it.
        """
        job_dir = self._paths.resolve_job(job_id)
        if not job_dir:
            raise ServiceError(404, "unknown job")
        if not LANG_RE.match(lang):
            raise ServiceError(404, "unknown lang")
        meta = self._repo.read_job_meta(job_dir)
        if not meta or lang not in meta.get("langs", []):
            raise ServiceError(404, "unknown job/lang")
        views: list[SegmentView] = []
        for chunk in self._repo.read_plan(job_dir).get("chunks", []):
            out_dir = self._paths.resolve_chunk_dir(job_dir, lang, chunk["idx"])
            if not out_dir or not out_dir.exists() or self._repo.read_status(out_dir).get("state") != "done":
                continue
            views += [self._view(chunk, seg) for seg in self._repo.load_segments(out_dir)]
        views.sort(key=lambda v: v.start)
        return SegmentList(segments=views)

    @staticmethod
    def _view(chunk: JsonObject, seg: Segment) -> SegmentView:
        history = seg.get("history")
        return SegmentView(
            chunk=chunk["idx"],
            id=seg.get("id"),
            start=chunk["start"] + seg.get("start", 0),
            end=chunk["start"] + seg.get("end", 0),
            speaker=seg.get("speaker"),
            gender=seg.get("gender"),
            src_text=seg.get("src_text"),
            mt_text=seg.get("mt_text"),
            tgt_text=seg.get("tgt_text"),
            edited=seg.get("edited", False),
            fallback=bool(seg.get("fallback", False)),
            fallback_reason=seg.get("fallback_reason"),
            history_len=len(history) if isinstance(history, list) else 0,
        )

    # ------------------------------------------------------------------ public edit operations
    async def edit(self, job_id: str, lang: str, chunk: int, seg_id: str, text: str) -> EditResponse:
        """Replace a segment's text and re-synthesise it.

        Raises:
            ServiceError: 422 for empty or too long text; 400/404/409/503 from :meth:`edit_flow`.
        """
        if not text or not text.strip():
            raise ServiceError(422, "text must not be empty")
        if len(text) > self._config.max_edit_chars:
            raise ServiceError(422, f"text too long ({len(text)} chars; the limit is {self._config.max_edit_chars})")
        return await self.edit_flow(job_id, lang, chunk, seg_id, lambda seg: apply_edit(seg, text))

    async def undo(self, job_id: str, lang: str, chunk: int, seg_id: str) -> EditResponse:
        """Undo the last edit of a segment and re-synthesise it.

        Raises:
            ServiceError: 409 when there is nothing to undo; otherwise as :meth:`edit_flow`.
        """
        remaining: list[Any] = []

        def change(seg: Segment) -> str:
            text, history = apply_undo(seg)
            remaining[:] = [history]
            return text

        def after(seg: Segment) -> None:
            # the pipeline must not own history: force our popped list back
            seg["history"] = remaining[0]
            seg["edited"] = bool(remaining[0])

        return await self.edit_flow(job_id, lang, chunk, seg_id, change, after)

    # ------------------------------------------------------------------ transactional flow
    def resolve_target(self, job_id: str, lang: str, chunk: int, seg_id: str) -> EditTarget:
        """Validate the path of an edit and locate its chunk.

        Raises:
            ServiceError: 404 for unknown job/lang/chunk, 400 for a malformed segment id.
        """
        job_dir = self._paths.resolve_job(job_id)
        if not job_dir:
            raise ServiceError(404, "unknown job")
        if not LANG_RE.match(lang):
            raise ServiceError(404, "unknown lang")
        if not SEG_RE.match(seg_id):
            raise ServiceError(400, "invalid seg id")
        meta = self._repo.read_job_meta(job_dir)
        if not meta or lang not in meta.get("langs", []):
            raise ServiceError(404, "unknown job/lang")
        planned = next((c for c in self._repo.read_plan(job_dir).get("chunks", []) if c["idx"] == chunk), None)
        if not planned:
            raise ServiceError(404, "chunk not found in plan")
        out_dir = self._paths.resolve_chunk_dir(job_dir, lang, chunk)
        if not out_dir or not out_dir.exists():
            raise ServiceError(404, "chunk dir not found")
        return EditTarget(job_dir, meta, planned, out_dir, lang, seg_id)

    async def edit_flow(
        self, job_id: str, lang: str, chunk: int, seg_id: str, make_change: Change, after: After | None = None
    ) -> EditResponse:
        """Shared edit/undo path.

        Under the chunk lock: take the gate (edit priority), snapshot ``segments.json`` byte-for-byte, apply the
        change, re-synthesise; on ANY failure restore the snapshot atomically so text, history and the ``edited`` flag
        are exactly as before. The published mp4 stays the previous version until the pipeline swaps it.

        Args:
            job_id: Job identifier.
            lang: Target language.
            chunk: Chunk index.
            seg_id: Segment identifier.
            make_change: Mutates the segment and returns the text to synthesise; may raise 409 before
                anything is touched.
            after: Optional fix-up applied to the segment the pipeline wrote back.

        Returns:
            The new chunk version and the stored segment.

        Raises:
            ServiceError: 404 (missing), 409 (chunk not done), 503 (server busy), 500 (re-synthesis failed).
        """
        target = self.resolve_target(job_id, lang, chunk, seg_id)
        async with self._locks.get(f"{job_id}:{lang}:{chunk}"):
            snapshot, segments, segment = self._load_for_edit(target.out_dir, seg_id)
            new_text = make_change(segment)
            await self._acquire_edit_slot(job_id, lang, chunk)
            try:
                await self._apply_transaction(target, new_text, snapshot, segments, after)
            finally:
                self._gate.release()
            version = self._commit(target.out_dir)
            _, stored = find_segment(self._repo.load_segments(target.out_dir), seg_id)
        return EditResponse(ok=True, version=version, segment=stored)

    def _load_for_edit(self, out_dir: Path, seg_id: str) -> tuple[bytes, Segments, Segment]:
        status = self._repo.read_status(out_dir)
        if status.get("state") != "done":
            raise ServiceError(409, f"chunk not done (state={status.get('state')})")
        try:
            snapshot = (out_dir / "segments.json").read_bytes()
        except OSError:
            raise ServiceError(404, "segments not found") from None
        segments = self._repo.load_segments(out_dir)
        _, segment = find_segment(segments, seg_id)
        if segment is None:
            raise ServiceError(404, "segment not found")
        return snapshot, segments, segment

    async def _acquire_edit_slot(self, job_id: str, lang: str, chunk: int) -> None:
        try:
            await self._gate.acquire(job_id, lang, chunk, edit=True, timeout=self._config.edit_gate_timeout_s)
        except (asyncio.TimeoutError, GateBusy):
            raise ServiceError(
                503, "the server is busy dubbing other parts; please retry the edit in a minute"
            ) from None

    async def _apply_transaction(
        self, target: EditTarget, text: str, snapshot: bytes, segments: Segments, after: After | None
    ) -> None:
        out_dir = target.out_dir
        pre_edit = out_dir / "segments.json.pre_edit"
        try:
            self._repo.write_bytes(pre_edit, snapshot)
            self._repo.update_status(out_dir, editing=True)
            self._repo.save_segments(out_dir, segments)
            await self._resynthesise(target, text)
            with contextlib.suppress(OSError):  # media is published: the edit is COMMITTED, drop the rollback snapshot
                pre_edit.unlink()
            if after is not None:
                self._run_after(out_dir, target.seg_id, after)
        except BaseException:
            self._roll_back(out_dir, snapshot)
            raise

    def _run_after(self, out_dir: Path, seg_id: str, after: After) -> None:
        segments = self._repo.load_segments(out_dir)
        _, segment = find_segment(segments, seg_id)
        if segment is not None:
            after(segment)
            self._repo.save_segments(out_dir, segments)

    def _roll_back(self, out_dir: Path, snapshot: bytes) -> None:
        try:
            self._repo.write_bytes(out_dir / "segments.json", snapshot)
        finally:
            with contextlib.suppress(OSError):
                (out_dir / "segments.json.pre_edit").unlink()
            with contextlib.suppress(OSError):
                self._repo.update_status(out_dir, editing=False)

    def _commit(self, out_dir: Path) -> int:
        with contextlib.suppress(OSError):
            (out_dir / "segments.json.pre_edit").unlink()
        version = int(self._repo.read_status(out_dir).get("version", 0) or 0) + 1
        self._repo.update_status(out_dir, version=version, editing=False)
        return version

    async def _resynthesise(self, target: EditTarget, text: str) -> None:
        source = target.job_dir / "source.mp4"
        url = str(source) if source.exists() else target.meta.get("url", "")
        args = [
            "--url",
            url,
            "--lang",
            target.lang,
            "--out",
            str(target.out_dir),
            "--resynth",
            target.seg_id,
            "--text",
            text,
        ]
        env = {"SPEAKER_MAP": str(target.job_dir / "speakers_global.json"), "CHUNK_START": str(target.chunk["start"])}
        rc, logs = await self._process.run(
            PIPELINE_MODULE,
            args,
            self._config.root,
            target.out_dir / "pipeline.log",
            extra_env=env,
            timeout=self._config.resynth_timeout_s,
            kind="resynth",
        )
        if rc != 0:
            raise ServiceError(500, f"resynth failed (exit {rc}): {logs[-500:]}")
