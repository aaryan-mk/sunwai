"""Voice-plan rollout: decided once per job at the first planner run, and the closure that computes the plan."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from sunwai.paths import DATA_DIR
from sunwai.planning.envvars import env_flag
from sunwai.planning.fileio import atomic_json, json_or_none
from sunwai.planning.joblock import job_lock
from sunwai.planning.speakers import PlanFn, SpeakerMap
from sunwai.planning.wavio import as_array, read_mono16k

log = logging.getLogger("planner")


def _language_codes() -> list[str]:
    """Target language codes from the packaged ``languages.json``."""
    config = json_or_none(str(DATA_DIR / "languages.json")) or {}
    return [target["code"] for target in config.get("targets", [])]


def _has_output_dir(out_dir: str, name: str, language_codes: set[str]) -> bool:
    """True for a pre-existing language dir or ``_prep`` (even empty), or any dir holding ``c*`` chunk entries."""
    full = os.path.join(out_dir, name)
    if not os.path.isdir(full) or name.startswith("."):
        return False
    return name in language_codes or name == "_prep" or any(c.startswith("c") for c in os.listdir(full))


def has_preexisting_output(out_dir: str) -> bool:
    """True when the job dir already holds a speaker map, a plan, or a language/chunk output directory."""
    codes = set(_language_codes())
    return (
        os.path.exists(os.path.join(out_dir, "speakers_global.json"))
        or os.path.exists(os.path.join(out_dir, "plan.json"))
        or any(_has_output_dir(out_dir, name, codes) for name in os.listdir(out_dir))
    )


def decide_voice_rollout(out_dir: str) -> bool:
    """Decide (and persist) whether this job gets a voice plan.

    Decided at the FIRST planner run of a job: rollout=1 ONLY if, at that moment, the job dir holds no
    ``speakers_global.json``, no ``plan.json`` from an earlier run and no language/chunk output directory (any name
    length, including ``_prep``); anything pre-existing is legacy for good. Retries read the marker and never
    re-decide.
    """
    marker_path = os.path.join(out_dir, "voice_rollout.json")
    with job_lock(out_dir):
        marker = json_or_none(marker_path)
        if isinstance(marker, dict) and "voice_plan" in marker:
            return bool(marker["voice_plan"] == 1)
        old = has_preexisting_output(out_dir)
        new = (not old) and env_flag("VOICE_PLAN") and env_flag("VOICE_XLING")
        decision = {
            "voice_plan": 1 if new else 0,
            "decided": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "preexisting": bool(old),
        }
        atomic_json(marker_path, decision, indent=2)
        log.info("[voice] rollout decided for this job: %s", "voice_plan" if new else "legacy")
        return new


def make_plan_fn(out_dir: str) -> PlanFn:
    """Closure that computes the voice plan for the finished speaker map (called before the single write)."""

    def plan(data: SpeakerMap) -> dict[str, Any]:
        from sunwai.planning import voices

        samples = as_array(read_mono16k(os.path.join(out_dir, "source16k.wav")))
        features = voices.speaker_features(samples, data["turns"], os.path.join(out_dir, ".ecapa"))
        langs = _language_codes()
        job = json_or_none(os.path.join(out_dir, "job.json")) or {}
        chosen = [lang for lang in (job.get("langs") or []) if lang in langs] or langs
        return voices.plan_job(data, langs, features, opt_langs=chosen)  # type: ignore[no-any-return]

    return plan
