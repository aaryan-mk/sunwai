"""Sync-quality metrics derived from the segments a chunk was dubbed with."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sunwai.server.repository import Segments
from sunwai.server.schemas import ChunkSync, LangSync


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def lipsync_is_fresh(out_dir: Path, lang: str) -> bool:
    """True when a non-empty lip-sync render exists and is newer than the aligned-audio dub."""
    dub, lip = out_dir / f"dub_{lang}.mp4", out_dir / f"dub_{lang}_lipsync.mp4"
    try:
        if lip.exists() and lip.stat().st_size > 0:
            return (not dub.exists()) or lip.stat().st_mtime > dub.stat().st_mtime
    except OSError:
        pass
    return False


def chunk_sync(segments: Segments, out_dir: Path, lang: str) -> ChunkSync:
    """Summarise one chunk: overflow, re-fits, tempo and lip-sync freshness.

    Args:
        segments: The chunk's segments.
        out_dir: Chunk directory (for the lip-sync file check).
        lang: Target language.

    Returns:
        The chunk's metrics.
    """
    tempos = [_as_float(s.get("tempo"), 1.0) for s in segments if s.get("tempo") is not None]
    return ChunkSync(
        segments=len(segments),
        overflowed=sum(1 for s in segments if s.get("overflow")),
        cut_s=round(sum(_as_float(s.get("overflow_s")) for s in segments), 2),
        refit=sum(1 for s in segments if s.get("refit")),
        tempo_mean=round(sum(tempos) / len(tempos), 3) if tempos else None,
        tempo_max=round(max(tempos), 3) if tempos else None,
        tempo_n=len(tempos),
        lipsync=lipsync_is_fresh(out_dir, lang),
    )


def lang_sync(chunks: Sequence[ChunkSync | None], done: int) -> LangSync:
    """Aggregate chunk metrics over the chunks that have segments.

    Args:
        chunks: Per-chunk metrics (``None`` for chunks without any).
        done: Chunks of this language that are finished.

    Returns:
        The language-level totals; ``tempo_mean`` is weighted by each chunk's tempo count.
    """
    have = [c for c in chunks if c is not None]
    weighted = [(c.tempo_mean, c.tempo_n) for c in have if c.tempo_mean is not None]
    tempo_max = [c.tempo_max for c in have if c.tempo_max is not None]
    lipsync_chunks = sum(1 for c in have if c.lipsync)
    weight = max(1, sum(w for _, w in weighted))
    return LangSync(
        chunks_measured=len(have),
        segments=sum(c.segments for c in have),
        overflowed=sum(c.overflowed for c in have),
        cut_s=round(sum(c.cut_s for c in have), 2),
        refit=sum(c.refit for c in have),
        tempo_mean=round(sum(m * w for m, w in weighted) / weight, 3) if weighted else None,
        tempo_max=max(tempo_max) if tempo_max else None,
        lipsync_chunks=lipsync_chunks,
        chunks_done=done,
        lipsync_coverage=round(lipsync_chunks / done, 3) if done else 0.0,
    )
