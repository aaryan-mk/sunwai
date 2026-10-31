"""Seams between the control plane and the expression and language-scheduler services.

Every seam is a module attribute holding a callable. The defaults raise :class:`NotImplementedHook` (HTTP 501) until
:mod:`sunwai.control.adapters` replaces them at registration time; callers always look the attribute up at call time.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Awaitable, Callable, Coroutine
from pathlib import Path
from typing import Any

Result = dict[str, Any]

ExpressionResynth = Callable[[Path, str, int, str, str | None, str, float | None, Path], Awaitable[Result]]
ExpressionApply = Callable[[Path, str, int, str, Result], Awaitable[Result]]
ExpressionUndo = Callable[[Path, str, int, str], Awaitable[Result]]
ApplyStart = Callable[[Path, str, int, str, Result], "concurrent.futures.Future[Result]"]
UndoStart = Callable[[Path, str, int, str], "concurrent.futures.Future[Result]"]
RequestLanguage = Callable[[str, str, int, int], Awaitable[Result]]
LanguageStatus = Callable[[str, str], Awaitable[Result]]
ReleaseLanguage = Callable[[str, str, str], None]
MediaPath = Callable[..., Awaitable[Path]]
LiveState = Callable[[Path, str, int, str], Result]


class NotImplementedHook(Exception):
    """Raised by a seam whose adapter is not installed."""


# One dedicated pool for expression mutations: the worker is tracked as a concurrent.futures.Future (NOT an asyncio Task),
# so no event-loop cancellation (wrapper, task or shutdown) can detach us from a thread that is still mutating live files.
EXEC = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="m2")


async def _resynth_stub(
    job_dir: Path, lang: str, chunk: int, segment_id: str, intent: str | None, base_text: str, tempo_factor: float | None, out_candidate_dir: Path
) -> Result:
    """Default for :data:`expression_resynth`: the expression pipeline is not installed."""
    raise NotImplementedHook("expression pipeline not installed")


async def _apply_stub(job_dir: Path, lang: str, chunk: int, segment_id: str, candidate: Result) -> Result:
    """Default for :data:`expression_apply`: the expression apply path is not installed."""
    raise NotImplementedHook("expression apply not installed")


async def _undo_stub(job_dir: Path, lang: str, chunk: int, segment_id: str) -> Result:
    """Default for :data:`expression_undo`: expression undo is not installed."""
    raise NotImplementedHook("expression undo not installed")


async def _request_language_stub(job_id: str, lang: str, from_ms: int, generation: int) -> Result:
    """Default for :data:`request_language`: the language scheduler is not installed."""
    raise NotImplementedHook("language scheduler not installed")


async def _language_status_stub(job_id: str, lang: str) -> Result:
    """Default for :data:`language_status`: the language scheduler is not installed."""
    raise NotImplementedHook("language status not installed")


async def _media_path_stub(
    job_dir: Path, lang: str, chunk: int, segment_id: str, kind: str, m2_candidate_id: str | None, is_head: bool = True, m2_revision: int | None = None
) -> Path:
    """Default for :data:`media_path`: no A/B audio is available."""
    raise NotImplementedHook("media not available")


def _live_state_stub(job_dir: Path, lang: str, chunk: int, segment_id: str) -> Result:
    """Default for :data:`live_state`: with no expression service nothing is owned."""
    return {"m2_head": None, "owned": False, "in_flight": False}


expression_resynth: ExpressionResynth = _resynth_stub
"""Synthesise a CANDIDATE line; never touches active audio. Result keys: ok, audio_hash, duration_ms, spoken_text, style, tempo_factor, error."""
expression_apply: ExpressionApply = _apply_stub
"""Remix the chunk with a candidate and queue lip sync. Result keys: ok, audio_hash, error."""
expression_undo: ExpressionUndo = _undo_stub
"""Undo a revision created by expression Apply."""
request_language: RequestLanguage = _request_language_stub
"""Language scheduler request. Result: ``{"track": {...}, "duplicate": bool}``."""
language_status: LanguageStatus = _language_status_stub
"""Track readiness: ``{lang, ready_intervals, audio_state(absent|preparing|partial|ready|error), lipsync_state}``."""
media_path: MediaPath = _media_path_stub
"""Path of the A/B audio (``cand`` = candidate preview, ``base`` = current line); must lie under the chunk directory."""
live_state: LiveState = _live_state_stub
"""Read-only, non-blocking expression state: ``{"m2_head", "owned", "in_flight"}``."""
release_language: ReleaseLanguage | None = None
"""Durable abandonment marker for a failed add-language operation; ``None`` until the scheduler adapter installs it."""


def _run_coro_in_thread(coro_fn: Callable[..., Coroutine[Any, Any, Result]], *args: Any) -> Result:
    return asyncio.run(coro_fn(*args))


def _expression_apply_start(job_dir: Path, lang: str, chunk: int, segment_id: str, candidate: Result) -> concurrent.futures.Future[Result]:
    """Default for :data:`expression_apply_start`: run :data:`expression_apply` on the mutation pool."""
    return EXEC.submit(_run_coro_in_thread, expression_apply, job_dir, lang, chunk, segment_id, candidate)


def _expression_undo_start(job_dir: Path, lang: str, chunk: int, segment_id: str) -> concurrent.futures.Future[Result]:
    """Default for :data:`expression_undo_start`: run :data:`expression_undo` on the mutation pool."""
    return EXEC.submit(_run_coro_in_thread, expression_undo, job_dir, lang, chunk, segment_id)


expression_apply_start: ApplyStart = _expression_apply_start
"""Start an apply on the mutation pool; returns a future resolving to the apply result dict."""
expression_undo_start: UndoStart = _expression_undo_start
"""Start an undo on the mutation pool; returns a future resolving to the undo result dict."""
