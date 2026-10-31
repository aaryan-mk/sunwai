"""Head reconciliation: keep the revision log in step with the live segment, without racing the expression service."""

from __future__ import annotations

from typing import Any, NamedTuple

from sunwai.control import hooks as ctl_hooks
from sunwai.control.codec import Record, seg_key, utcnow
from sunwai.control.errors import CtlError
from sunwai.control.store import JobStore


class HeadView(NamedTuple):
    """The reconciled head of a segment together with what produced it."""

    head: Record
    seg: Record
    chunk_start: float
    store: JobStore
    key: str


def spoken_text(seg: Record) -> str:
    """Return the actually spoken text: the expression ``tts_text`` when it belongs to the current display text, else the display text."""
    text = seg.get("tgt_text", "")
    return seg["tts_text"] if seg.get("tts_text") and seg.get("tts_src") == text else text  # type: ignore[no-any-return]


def history_length(seg: Record) -> int:
    """Return the number of entries in the segment's edit history (0 when absent)."""
    return len(seg["history"]) if isinstance(seg.get("history"), list) else 0


def diverged(head: Record, seg: Record) -> bool:
    """Tell whether the live segment no longer matches the head revision."""
    if head["display_text"] != seg.get("tgt_text", "") or head.get("spoken_text") != spoken_text(seg):
        return True
    return head.get("live_hist") is not None and head["live_hist"] != history_length(seg)


def reconciled_revision(
    job_id: str,
    lang: str,
    chunk: int,
    seg_id: str,
    seg: Record,
    chunk_start: float,
    head: Record | None,
    audio_hash: str,
) -> Record:
    """Build the revision that adopts the live segment: ``base`` (rev 0) or ``legacy`` (an old-route edit)."""
    provenance = "base" if head is None else "legacy"
    return {
        "job_id": job_id,
        "lang": lang,
        "chunk_id": int(chunk),
        "segment_id": seg_id,
        "parent_revision": head["revision"] if head else None,
        "source_revision": 0,
        "speaker_id": seg.get("speaker"),
        "voice_plan_id": None,
        "slot_start_ms": int((chunk_start + seg.get("start", 0)) * 1000),
        "slot_end_ms": int((chunk_start + seg.get("end", 0)) * 1000),
        "raw_source_text": seg.get("src_text", ""),
        "display_text": seg.get("tgt_text", ""),
        "spoken_text": spoken_text(seg),
        "live_hist": history_length(seg),
        "expression": {"intent": None, "emotion": None, "confidence": None, "provenance": provenance},
        "style": None,
        "tempo_factor": 1.0,
        "audio_hash": audio_hash,
        "subtitle_revision": (head or {}).get("subtitle_revision", 0),
        "status": "active",
        "ts": utcnow(),
    }


def find_segment(deps: Any, job_dir: Any, lang: str, chunk: int, seg_id: str) -> tuple[Record, float]:
    """Locate a live segment through the host.

    Raises:
        CtlError: 404 for an unknown language, chunk or segment.
    """
    if lang not in deps.job_langs(job_dir):
        raise CtlError(404, "unknown_lang", "language is not part of this job")
    if int(chunk) not in deps.chunk_ids(job_dir):
        raise CtlError(404, "unknown_chunk", "unknown chunk")
    segments, chunk_start = deps.segments(job_dir, lang, int(chunk))
    for seg in segments:
        if str(seg.get("id")) == seg_id:
            return seg, chunk_start
    raise CtlError(404, "unknown_segment", "segment not found")


def reconcile_head(
    deps: Any, job_dir: Any, job_id: str, store: JobStore, lang: str, chunk: int, seg_id: str
) -> HeadView:
    """Reconcile the live segment into the revision log (a rev-0 base, or a ``legacy`` revision after an old-route edit).

    A divergence is NEVER reconciled into a ``legacy`` revision while an operation of ours has the line claimed, while an
    expression apply/undo holds the line right now, or when the live line equals the expression service's own head state:
    that race once produced a spurious revision and "committed but not recorded".
    """
    seg, chunk_start = find_segment(deps, job_dir, lang, chunk, seg_id)
    key = seg_key(lang, chunk, seg_id)
    with store.lock:
        head = store.head(key)
        if head is None or diverged(head, seg):
            if head is not None:
                live = ctl_hooks.live_state(job_dir, lang, chunk, seg_id)
                if key in store.running or live.get("in_flight") or live.get("owned"):
                    return HeadView(head, seg, chunk_start, store, key)
            audio_hash = deps.audio_hash(job_dir, lang, chunk, seg.get("tgt_text", ""))
            head = store.add_revision(
                key, reconciled_revision(job_id, lang, chunk, seg_id, seg, chunk_start, head, audio_hash)
            )
    return HeadView(head, seg, chunk_start, store, key)
