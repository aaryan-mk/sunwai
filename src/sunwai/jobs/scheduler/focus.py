"""Mapping of a new track's start onto the server's single-slot ``/focus`` priority."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sunwai.jobs.scheduler import store
from sunwai.jobs.scheduler.errors import Hooks
from sunwai.jobs.scheduler.ordering import chunk_for_ms, runway


def focus_adapter(
    job_dir: Path, job_id: str, watching_lang: str | None, lang: str, from_ms: float, hooks: Hooks
) -> dict[str, Any]:
    """Focus the new language at the chunk containing ``from_ms``, once the viewer's own runway is buffered.

    The viewer's current language keeps priority until its runway is buffered. Focus is advisory: failures are
    reported in the result, never raised.

    Returns:
        ``{"applied": bool, "chunk": int, "reason": str}``.
    """
    chunks = store.job_chunks(job_dir)
    k = chunk_for_ms(chunks, from_ms)
    if hooks.set_focus is None:
        return {"applied": False, "chunk": k, "reason": "no_focus_hook"}
    if watching_lang and watching_lang != lang:
        pending = [i for i in runway(chunks, from_ms) if not store.chunk_done(job_dir, watching_lang, i)]
        if pending:
            return {"applied": False, "chunk": k, "reason": "watching_lang_runway_not_buffered"}
    try:
        hooks.set_focus(job_id, lang, k)
    except Exception as exc:
        return {"applied": False, "chunk": k, "reason": f"set_focus_failed:{type(exc).__name__}"}
    return {"applied": True, "chunk": k, "reason": "ok"}
