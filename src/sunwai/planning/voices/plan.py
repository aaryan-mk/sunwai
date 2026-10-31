"""Build the immutable per-job voice plan and validate it again on the worker side."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from sunwai.paths import DATA_DIR
from sunwai.planning.voices.assignment import VoicePlanner
from sunwai.planning.voices.catalogue import PLAN_VERSION, catalogue_rev, load_bank, smap_hash
from sunwai.planning.voices.gate import cer_gate, eligible
from sunwai.planning.voices.types import Bank, Features, PlanError, Speaker, VoiceAssignment

AssignFn = Callable[[list[Speaker], dict[Any, Features | None], list[str], Bank, list[str] | None], VoiceAssignment]


def assign_voices(
    speakers: list[Speaker],
    feats: dict[Any, Features | None],
    langs: list[str],
    bank: Bank,
    opt_langs: list[str] | None = None,
) -> VoiceAssignment:
    """Assign voices with the default :class:`VoicePlanner` (the pluggable assignment function of :func:`plan_job`)."""
    return VoicePlanner(bank, langs, opt_langs).assign(speakers, feats)


def plan_job(
    speakers_global: dict[str, Any],
    langs: list[str],
    feats: dict[Any, Features | None],
    bank: Bank | None = None,
    assign: AssignFn = assign_voices,
    app: Path = DATA_DIR,
    opt_langs: list[str] | None = None,
) -> dict[str, Any]:
    """Build the immutable ``voice_plan`` for the ``speakers_global.json`` content (without a plan).

    Raises:
        PlanError: The voice bank is missing or there are too few gated voices.
    """
    bank = bank or load_bank(app)
    if not bank.get("voices"):
        raise PlanError("voice bank missing")
    speakers = [
        {"id": x["id"], "gender": x.get("gender"), "p_female": x.get("p_female"), "total_s": x.get("total_s")}
        for x in speakers_global.get("speakers", [])
    ]
    result = assign(speakers, feats, list(langs), bank, opt_langs)
    return {
        "version": PLAN_VERSION,
        "catalogue_rev": catalogue_rev(app),
        "smap_hash": smap_hash(speakers_global),
        "langs": list(langs),
        "canonical": result.canonical,
        "voices": result.per_language,
        "distinctness_degraded": bool(result.degraded),
        "tau": bank["tau"],
        "unmeasured_langs": [lang for lang in langs if lang not in bank["langs"]],
        "unavailable_langs": dict(result.unavailable),
        "cer_gate": cer_gate(bank),
        "warnings": result.warnings,
        "speakers": result.detail,
    }


def _checked_header(plan: Any, speakers_global: dict[str, Any], lang: str, app: Path) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise PlanError("voice_plan is not an object")
    if plan.get("version") != PLAN_VERSION:
        raise PlanError(f"unsupported voice_plan version {plan.get('version')!r}")
    if plan.get("smap_hash") != smap_hash(speakers_global):
        raise PlanError("voice_plan smap_hash does not match the speaker map")
    rev = catalogue_rev(app)
    if plan.get("catalogue_rev") != rev:
        raise PlanError(f"voice_plan catalogue_rev {plan.get('catalogue_rev')} != {rev}")
    if lang not in (plan.get("langs") or []):
        raise PlanError(f"language {lang} not in voice_plan.langs")
    if lang in (plan.get("unavailable_langs") or {}):
        raise PlanError(f"language {lang} is unavailable for this job: {plan['unavailable_langs'][lang]}")
    return plan


def validate_plan(speakers_global: dict[str, Any], lang: str, app: Path = DATA_DIR) -> dict[str, Any]:
    """Preflight a chunk against its job's plan (worker side, read-only).

    Raises:
        PlanError: For ANY problem of a flagged job (stale catalogue, edited speaker map, gate failure, collision).

    Returns:
        The validated plan.
    """
    plan = _checked_header(speakers_global.get("voice_plan"), speakers_global, lang, app)
    voice_map = (plan.get("voices") or {}).get(lang)
    if not isinstance(voice_map, dict):
        raise PlanError(f"voice_plan has no mapping for {lang}")
    bank = load_bank(app)
    known = set(bank.get("voices", {}))
    gate = cer_gate(bank)
    unmeasured = set(plan.get("unmeasured_langs") or [])
    if (lang in bank.get("langs", {})) == (lang in unmeasured):
        raise PlanError(f"{lang}: measured/unmeasured flag inconsistent with the bank")
    used: list[str] = []
    for speaker in speakers_global.get("speakers", []):
        voice = voice_map.get(str(speaker["id"]))
        if not isinstance(voice, str) or voice not in known:
            raise PlanError(f"speaker {speaker['id']} has no valid voice for {lang}: {voice!r}")
        if lang in unmeasured:
            if not all(eligible(bank, gate, m, voice) for m in bank["langs"]):
                raise PlanError(f"unmeasured language {lang}: {voice} fails the CER gate in a measured language")
        elif not eligible(bank, gate, lang, voice):
            raise PlanError(f"voice {voice} fails the CER gate in {lang}")
        used.append(voice)
    if len(set(used)) != len(used):
        raise PlanError(f"voice collision in {lang}: {used}")
    return plan
