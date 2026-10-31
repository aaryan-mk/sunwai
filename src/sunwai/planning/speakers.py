"""Stage 7 -- assemble ``speakers_global.json`` (speakers, gender labels, turns and, for new jobs, the voice plan)."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from typing import Any

from sunwai.planning.artifacts import SPEAKER_MAP
from sunwai.planning.errors import PlannerError
from sunwai.planning.fileio import atomic_json, json_or_none
from sunwai.planning.gender import assign_gender
from sunwai.planning.joblock import job_lock
from sunwai.planning.speaker_turns import Turn

log = logging.getLogger("planner")

SpeakerMap = dict[str, Any]
PlanFn = Callable[[SpeakerMap], dict[str, Any]]


def _gender_record(gender: dict[Any, Any], spk: Any) -> dict[str, Any]:
    """Look a speaker up in the gender stats, tolerating str-keyed (JSON-reloaded) dicts from any caller."""
    record = gender.get(spk)
    if record is None and os.environ.get("GENDER_KEYFIX", "1") == "1":
        record = gender.get(str(spk))
    return record if isinstance(record, dict) else {"mean_p": 0.5}


def build_speaker_map(
    backend: str,
    turns: Sequence[Turn],
    gender: dict[Any, Any],
    degraded: bool | None = None,
    degraded_reason: str | None = None,
) -> SpeakerMap:
    """Build the ``speakers_global.json`` document (without a voice plan).

    Args:
        backend: Diarization backend name.
        turns: Final speaker turns.
        gender: Per-speaker gender statistics (``mean_p`` is the mean female probability).
        degraded: Whether the diarization fell back to ECAPA; inferred from ``backend`` when ``None``.
        degraded_reason: Why it degraded.
    """
    by_speaker: dict[Any, list[Turn]] = {}
    for turn in turns:
        by_speaker.setdefault(turn["speaker"], []).append(turn)
    speakers = []
    for spk in sorted(by_speaker):
        spk_turns = by_speaker[spk]
        total_s = sum(t["end"] - t["start"] for t in spk_turns)
        mean_p = _gender_record(gender, spk).get("mean_p", 0.5)
        speakers.append(
            {
                "id": spk,
                "gender": assign_gender(mean_p, total_s, len(spk_turns)),
                "p_female": round(mean_p, 4),
                "total_s": round(total_s, 3),
                "n_turns": len(spk_turns),
            }
        )
    if degraded is None:
        degraded = str(backend).startswith("ecapa")
    return {
        "backend": backend,
        "degraded": bool(degraded),
        "degraded_reason": degraded_reason if degraded else None,
        "speakers": sorted(speakers, key=lambda s: s["id"]),
        "turns": list(turns),
    }


def _attach_voice_plan(data: SpeakerMap, previous: Any, plan_fn: PlanFn) -> None:
    """Add the voice plan to ``data``: an existing plan is immutable, a missing one is computed (and may fail)."""
    from sunwai.planning import voices

    existing = previous.get("voice_plan") if isinstance(previous, dict) else None
    if isinstance(existing, dict):
        if existing.get("smap_hash") != voices.smap_hash(data):
            raise PlannerError(
                "voice plan is immutable but the speaker map changed on retry (smap_hash "
                f"{existing.get('smap_hash')} -> {voices.smap_hash(data)}); old plan and files kept"
            )
        data["voice_plan"] = existing
        return
    try:
        data["voice_plan"] = plan_fn(data)
    except Exception as exc:
        raise PlannerError(f"voice plan failed for a rollout=1 job: {type(exc).__name__}: {exc}") from exc


def stage_write(
    out_dir: str,
    backend: str,
    turns: Sequence[Turn],
    gender: dict[Any, Any],
    degraded: bool | None = None,
    degraded_reason: str | None = None,
    plan_fn: PlanFn | None = None,
) -> SpeakerMap:
    """Publish ``speakers_global.json`` in ONE atomic write of the complete contract, under the job lock.

    For a rollout=1 job the voice plan is computed BEFORE the file is written and is part of the same write; an
    existing plan is immutable (a changed map hard-fails with the files untouched); a planning failure FAILS the
    planner instead of silently downgrading to legacy.
    """
    data = build_speaker_map(backend, turns, gender, degraded, degraded_reason)
    path = os.path.join(out_dir, SPEAKER_MAP)
    with job_lock(out_dir):
        previous = json_or_none(path)
        if plan_fn is not None:
            if previous is None and os.path.exists(path):
                raise PlannerError(
                    "speakers_global.json exists but is unreadable on a rollout=1 job; refusing to overwrite it"
                )
            _attach_voice_plan(data, previous, plan_fn)
        atomic_json(path, data, indent=2, ensure_ascii=False)
        plan = data.get("voice_plan")
        if plan:
            log.info(
                "[voice] plan v%s degraded=%s canonical=%s warnings=%s",
                plan.get("version"),
                plan.get("distinctness_degraded"),
                plan.get("canonical"),
                plan.get("warnings"),
            )
    return data
