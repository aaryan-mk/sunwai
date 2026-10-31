"""Voice resolution: which voice a line is re-synthesised with. The immutable voice plan always wins."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sunwai.control.expression import engine_bridge
from sunwai.control.expression.errors import ExprError
from sunwai.control.expression.fsutil import read_json


def is_planned_job(rollout: Any, speakers: Any) -> bool:
    """Tell whether a job uses an immutable voice plan (rollout flag or a plan embedded in the speaker map)."""
    flagged = isinstance(rollout, dict) and rollout.get("voice_plan") == 1
    return flagged or (isinstance(speakers, dict) and "voice_plan" in speakers)


def planned_voice(speakers: Any, lang: str, speaker: Any) -> str:
    """Return the plan's voice for ``speaker`` in ``lang``.

    Raises:
        ExprError: 422 when the plan is missing, invalid, or has no voice for the speaker (no legacy fallback).
    """
    if not isinstance(speakers, dict) or "voice_plan" not in speakers:
        raise ExprError(422, "planned job but speakers_global.json has no voice_plan; refusing legacy voice")
    try:
        from sunwai.planning import voices as plan_voices

        voice = plan_voices.validate_plan(speakers, lang)["voices"][lang].get(str(speaker))
    except Exception as exc:
        raise ExprError(422, f"voice plan invalid: {exc}") from exc
    if not voice:
        raise ExprError(422, f"voice plan has no voice for speaker {speaker}")
    return voice  # type: ignore[no-any-return]


def resolve_voice(out_dir: str | Path, lang: str, seg: dict[str, Any]) -> str:
    """Resolve the voice for a segment.

    A planned job (``voice_rollout.json`` with ``voice_plan == 1`` or a ``voice_plan`` in the speaker map) gets the plan's
    voice for this speaker, with an error instead of a legacy fallback. A legacy job uses ``seg["voice"]`` or the engine's
    ``pick_voice``.
    """
    job_dir = Path(out_dir).resolve().parent.parent
    speakers = read_json(os.environ.get("SPEAKER_MAP") or str(job_dir / "speakers_global.json"))
    rollout = read_json(job_dir / "voice_rollout.json") or {}
    if is_planned_job(rollout, speakers):
        return planned_voice(speakers, lang, seg.get("speaker"))
    if seg.get("voice"):
        return seg["voice"]  # type: ignore[no-any-return]
    return engine_bridge.pick_voice(lang, seg["gender"], seg["speaker"])
