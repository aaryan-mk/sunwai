"""Pure validation and bookkeeping rules for add-language requests (no filesystem access)."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any

from sunwai.jobs.scheduler.config import orphan_grace_s
from sunwai.jobs.scheduler.errors import LangSchedError

LANG_RE = re.compile(r"^[a-z]{2,3}$")


@dataclass(frozen=True)
class TrackRequest:
    """A validated add-language request.

    Attributes:
        lang: Normalised target language code.
        from_ms: Playhead position (ms) the viewer switched at.
        generation: Playback generation; older than the newest seen is rejected.
    """

    lang: str
    from_ms: int
    generation: int


def parse_request(lang: Any, from_ms: Any, playback_generation: Any) -> TrackRequest:
    """Validate and normalise raw request arguments.

    Raises:
        LangSchedError: 422 ``bad_lang`` or ``bad_args`` when an argument is malformed.
    """
    code = lang.strip().lower() if isinstance(lang, str) else ""
    if not LANG_RE.match(code):
        raise LangSchedError(422, "bad_lang", "invalid language code")
    try:
        position = float(from_ms)
        generation = int(playback_generation)
    except (TypeError, ValueError):
        raise LangSchedError(422, "bad_args", "from_ms must be a number and playback_generation an integer") from None
    if not math.isfinite(position) or position < 0 or generation < 0:
        raise LangSchedError(422, "bad_args", "from_ms >= 0 and playback_generation >= 0 required")
    return TrackRequest(code, int(position), generation)


def voice_plan_verdict(speakers_global: Any, lang: str) -> tuple[bool, str]:
    """Decide whether a job's frozen voice plan admits ``lang``.

    A planned job accepts only ``voice_plan.langs`` minus ``unavailable_langs``; a legacy job (no plan) is accepted.

    Args:
        speakers_global: Parsed ``speakers_global.json``.
        lang: Requested language.

    Returns:
        ``(ok, reason)``.
    """
    plan = speakers_global.get("voice_plan") if isinstance(speakers_global, dict) else None
    if plan is None:
        return True, "legacy"
    if not isinstance(plan, dict):
        return False, "voice_plan_invalid"
    unavailable = plan.get("unavailable_langs") or {}
    if lang in unavailable:
        return False, f"voice plan marks {lang} unavailable: {unavailable[lang]}"
    if lang not in (plan.get("langs") or []) or lang not in (plan.get("voices") or {}):
        return False, f"{lang} is not in this job's immutable voice plan"
    return True, "planned"


def voice_plan_error(reason: str) -> LangSchedError:
    """Map a failed :func:`voice_plan_verdict` reason to the API error."""
    if reason == "planner_not_ready":
        return LangSchedError(409, "planner_not_ready", "planner has not finished; retry shortly")
    return LangSchedError(
        422,
        "voice_plan_excludes_lang",
        f"{reason}. Voices are frozen per job: start a NEW job with this language to get a voice plan for it.",
    )


def releases_cap(track: dict[str, Any], job_alive: bool | None, now: float | None = None) -> bool:
    """Return True when an unfinished track no longer counts toward the global in-flight cap.

    * ``abandoned`` (durable marker from a release): frees once the job's task is not known alive, so work started
      after the marker still counts and a late release needs no retry.
    * ``scheduled`` with no live job task: frees only after ``DUB_ADDLANG_ORPHAN_S`` (default 1800 s).
    * ``scheduling`` (uncommitted) never frees: it must be recovered.

    Args:
        track: The track record from ``tracks.json``.
        job_alive: True/False when the job's task liveness is known, None when unknown (no hook).
        now: Current time, injectable for tests.
    """
    phase = track.get("phase")
    if phase == "abandoned":
        return job_alive is not True
    if phase == "scheduled" and job_alive is False:
        age = (time.time() if now is None else now) - float(track.get("created") or 0)
        return age > orphan_grace_s()
    return False


def new_intent(req: TrackRequest, watching_lang: str | None, prev_langs: list[str], now: float) -> dict[str, Any]:
    """Return the write-ahead ``scheduling`` record persisted before any job mutation."""
    return {
        "from_ms": req.from_ms,
        "generation": req.generation,
        "created": now,
        "requests": 1,
        "watching_lang": watching_lang,
        "phase": "scheduling",
        "prev_langs": prev_langs,
    }


def touched_record(
    prev: dict[str, Any] | None, req: TrackRequest, watching_lang: str | None, now: float
) -> dict[str, Any]:
    """Return the record for a repeat request of an existing track (counts it, keeps creation time and phase)."""
    record: dict[str, Any] = {
        "from_ms": req.from_ms,
        "generation": req.generation,
        "created": (prev or {}).get("created") or now,
        "requests": int((prev or {}).get("requests", 0)) + 1,
        "watching_lang": watching_lang,
    }
    if prev and prev.get("phase"):
        record["phase"] = prev["phase"]
    return record
