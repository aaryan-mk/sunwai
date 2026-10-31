"""What of an added-language track is playable: per-chunk audio and lip-sync state, and the summary enums."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sunwai.jobs.scheduler import store
from sunwai.jobs.scheduler.errors import Hooks
from sunwai.jobs.scheduler.ordering import merge_intervals
from sunwai.jobs.scheduler.reconcile import LEDGER


def audio_state(n_chunks: int, n_done: int, n_running: int, n_error: int) -> str:
    """Summarise chunk counts into the contract enum ``ready|partial|error|preparing``.

    ``queued`` chunks count as ``preparing``; ``error`` is reported only when nothing is still running.
    """
    if n_chunks and n_done == n_chunks:
        return "ready"
    if n_done:
        return "partial"
    if n_error and not n_running:
        return "error"
    return "preparing"


def lipsync_state(n_chunks: int, n_lipsynced: int) -> str:
    """Summarise lip-sync progress into ``none|partial|ready``."""
    if not n_lipsynced:
        return "none"
    return "ready" if n_lipsynced == n_chunks else "partial"


def chunk_lipsync(job_dir: Path, lang: str, idx: int, hook: Any = None) -> tuple[bool, str]:
    """Decide whether a chunk's lip-synced video is usable.

    The lip-synced file must be newer than the dub; then the audio hash is checked: ``hook`` if given, else
    ``lipsync.json``'s ``chunk_audio_hash`` against ``sha256(dub.wav)`` when recorded. A False hash is not ready.

    Args:
        job_dir: Job directory.
        lang: Target language.
        idx: Chunk index.
        hook: Optional ``(job_dir, lang, idx) -> bool | None`` audio-hash verdict.

    Returns:
        ``(ready, basis)`` where basis ``mtime`` means the hash could not be verified.
    """
    d = store.chunk_dir(job_dir, lang, idx)
    dub, synced = d / f"dub_{lang}.mp4", d / f"dub_{lang}_lipsync.mp4"
    try:
        fresh = synced.stat().st_size > 0 and dub.exists() and synced.stat().st_mtime > dub.stat().st_mtime
    except OSError:
        fresh = False
    if not fresh:
        return False, "stale_or_missing"
    verdict: bool | None = None
    if hook is not None:
        try:
            verdict = hook(Path(job_dir), lang, idx)
        except Exception:  # an advisory hook must never break readiness
            verdict = None
    if verdict is None:
        want = store.read_json(d / "lipsync.json").get("chunk_audio_hash")
        if want:
            verdict = store.file_sha256(d / "dub.wav") == want
    if verdict is False:
        return False, "hash_mismatch"
    return True, "hash" if verdict else "mtime"


def track_readiness(job_dir: Path, lang: str, hooks: Hooks | None = None) -> dict[str, Any]:
    """Report which intervals of ``lang`` are playable (audio and lip-sync) plus any pending reconcile note."""
    job_dir = Path(job_dir)
    meta = store.read_json(job_dir / "job.json")
    chunks = store.job_chunks(job_dir)
    if lang not in (meta.get("langs") or []):
        return {
            "lang": lang,
            "ready_intervals": [],
            "audio_state": "absent",
            "lipsync_state": "none",
            "lipsync_intervals": [],
            "chunks": [],
        }
    audio_iv: list[list[int]] = []
    sync_iv: list[list[int]] = []
    rows: list[dict[str, Any]] = []
    n_done = n_sync = n_run = n_err = 0
    hook = hooks.lipsync_hash_ok if hooks else None
    for c in chunks:
        state = store.read_json(store.chunk_dir(job_dir, lang, c["idx"]) / "status.json").get("state")
        done = store.chunk_done(job_dir, lang, c["idx"])
        synced, basis = chunk_lipsync(job_dir, lang, c["idx"], hook) if done else (False, "no_audio")
        interval = [int(round(c["start"] * 1000)), int(round((c["start"] + c["dur"]) * 1000))]
        n_done += done
        n_sync += synced
        n_run += not done and state in ("running", "queued")
        n_err += not done and state == "error"
        if done:
            audio_iv.append(interval)
        if synced:
            sync_iv.append(interval)
        rows.append(
            {
                "chunk": c["idx"],
                "state": "done" if done else (state or "pending"),
                "lipsync": bool(synced),
                "lipsync_basis": basis,
            }
        )
    verified = bool(n_sync) and all(r["lipsync_basis"] == "hash" for r in rows if r["lipsync"])
    return {
        "lang": lang,
        "ready_intervals": merge_intervals(audio_iv),
        "audio_state": audio_state(len(chunks), n_done, n_run, n_err),
        "lipsync_state": lipsync_state(len(chunks), n_sync),
        "lipsync_intervals": merge_intervals(sync_iv),
        "reconcile": LEDGER.get(job_dir, lang),
        "lipsync_verified": verified,
        "chunks": rows,
    }
