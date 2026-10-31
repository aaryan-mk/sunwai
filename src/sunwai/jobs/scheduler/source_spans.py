"""Per-job source-language spans (code-switching records) and reviewer overrides."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from sunwai.jobs.scheduler import store
from sunwai.jobs.scheduler.errors import LangSchedError
from sunwai.jobs.scheduler.requests import LANG_RE
from sunwai.jobs.scheduler.script_lid import classify_text


def _seg_key(chunk: int, seg_id: Any) -> str:
    return f"{int(chunk)}:{seg_id}"


def _job_lid(job_dir: Path) -> str | None:
    data = store.read_json(Path(job_dir) / "lid.json")
    code = data.get("code") or (str(data.get("label", "")).split(":")[0].strip() or None)
    return code or None


def _source_segments(job_dir: Path, idx: int, langs: list[str]) -> list[dict[str, Any]]:
    """Return cached source ASR: shared ``_prep`` first, else any language dir's segments (text is language-free)."""
    candidates = [Path(job_dir) / "_prep" / f"c{idx:04d}" / "segments.json"]
    candidates += [store.chunk_dir(job_dir, lang, idx) / "segments.json" for lang in langs]
    for path in candidates:
        segments = store.read_json(path, default=[])
        if isinstance(segments, dict):
            segments = segments.get("segments", [])
        if isinstance(segments, list) and segments:
            return segments
    return []


def build_source_spans(job_dir: Path) -> list[dict[str, Any]]:
    """Classify every cached source segment of a job, applying reviewer overrides."""
    job_dir = Path(job_dir)
    langs = list(store.read_json(job_dir / "job.json").get("langs") or [])
    lid = _job_lid(job_dir)
    overrides = store.read_json(job_dir / "source_lang_overrides.json")
    spans: list[dict[str, Any]] = []
    for chunk in store.job_chunks(job_dir):
        for seg in _source_segments(job_dir, chunk["idx"], langs):
            try:
                start = int(round((chunk["start"] + float(seg["start"])) * 1000))
                end = int(round((chunk["start"] + float(seg["end"])) * 1000))
            except (KeyError, TypeError, ValueError):
                continue
            verdict = classify_text(seg.get("src_text") or "", lid)
            override = overrides.get(_seg_key(chunk["idx"], seg.get("id")))
            if override:
                verdict.update(lang=override["lang"], confidence=1.0, method="override")
            spans.append(
                {
                    "start_ms": start,
                    "end_ms": end,
                    "lang": verdict["lang"],
                    "confidence": verdict["confidence"],
                    "method": verdict["method"],
                    "chunk": chunk["idx"],
                    "seg_id": seg.get("id"),
                    "mixed": verdict["mixed"],
                    "embedded": verdict["embedded"],
                }
            )
    return spans


def persist_source_spans(job_dir: Path) -> list[dict[str, Any]]:
    """Rebuild and atomically write ``source_spans.json``; return the spans."""
    spans = build_source_spans(job_dir)
    with store.LOCK:
        store.atomic_json(Path(job_dir) / "source_spans.json", {"job_lid": _job_lid(job_dir), "spans": spans})
    return spans


def load_source_spans(job_dir: Path) -> list[dict[str, Any]]:
    """Return the persisted spans (empty when none were written)."""
    data = store.read_json(Path(job_dir) / "source_spans.json")
    return data.get("spans", []) if isinstance(data, dict) else []  # type: ignore[no-any-return]


def set_segment_lang_override(
    job_dir: Path, chunk: int, seg_id: Any, lang: str, reviewer: str = ""
) -> list[dict[str, Any]]:
    """Record a reviewer's language for one segment and refresh the spans.

    Raises:
        LangSchedError: 422 ``bad_lang`` for a code that is neither a language nor ``und``.
    """
    if not isinstance(lang, str) or not (LANG_RE.match(lang) or lang == "und"):
        raise LangSchedError(422, "bad_lang", "invalid language code")
    with store.LOCK:
        path = Path(job_dir) / "source_lang_overrides.json"
        overrides = store.read_json(path)
        overrides[_seg_key(chunk, seg_id)] = {"lang": lang, "by": reviewer, "ts": time.time()}
        store.atomic_json(path, overrides)
    return persist_source_spans(job_dir)


def clear_segment_lang_override(job_dir: Path, chunk: int, seg_id: Any) -> list[dict[str, Any]]:
    """Remove a reviewer override and refresh the spans."""
    with store.LOCK:
        path = Path(job_dir) / "source_lang_overrides.json"
        overrides = store.read_json(path)
        overrides.pop(_seg_key(chunk, seg_id), None)
        store.atomic_json(path, overrides)
    return persist_source_spans(job_dir)
