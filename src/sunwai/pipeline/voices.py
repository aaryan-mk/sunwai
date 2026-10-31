"""Voice selection: language voice pools, voice plan preflight and per-speaker voice picking."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple


from sunwai.paths import DATA_DIR

log = logging.getLogger(__name__)




LANGS = {"hi", "ta", "mr", "ur", "bn", "te"}


try:
    LANGS |= {t["code"] for t in json.load(open(DATA_DIR / "languages.json", encoding="utf-8"))["targets"]}
except Exception:
    pass


# Voice map
VOICE_MAP: Dict[Tuple[str, str], str] = {
    ("hi", "male"): "Amit",
    ("hi", "female"): "Kavya",
    ("ta", "male"): "Arun",
    ("ta", "female"): "Anitha",
    # others fall back to Amit/Kavya for male/female
}


DEFAULT_VOICE = {"male": "Amit", "female": "Kavya", "unknown": "Amit"}


# ---------------------------------------------------------------------------
# Voice selection
# ---------------------------------------------------------------------------
_POOLS = None


def _voice_pools(lang: str):
    global _POOLS
    if _POOLS is None:
        try:
            cfg = json.load(open(DATA_DIR / "languages.json", encoding="utf-8"))
            _POOLS = {t["code"]: t.get("voices", {}) for t in cfg.get("targets", [])}
        except Exception:
            _POOLS = {}
    return _POOLS.get(lang, {})


# the voice plan is computed by the PLANNER (new jobs only) and embedded in speakers_global.json as "voice_plan".
# Workers only read it: validated once at preflight (a flagged job with a corrupt/incomplete/mismatched plan FAILS the chunk,
# there is no fallback to the legacy rule inside a planned job). No "voice_plan" key = legacy job = the exact old pick_voice.
_PLANNED_VOICES: Optional[Dict[str, str]] = None      # speaker id (str) -> voice for args.lang; None = legacy job


def voice_preflight(lang: str, out_dir: Optional[Path] = None) -> None:
    """Planned jobs are detected independently of SPEAKER_MAP (job-level rollout marker OR the map's own flag): a planned job
    whose map is missing/unreadable/unflagged FAILS. A job with neither marker nor flag is legacy."""
    global _PLANNED_VOICES
    _PLANNED_VOICES = None
    smap = os.environ.get("SPEAKER_MAP")
    jobdirs = []
    if smap: jobdirs.append(Path(os.path.dirname(smap)))
    if out_dir is not None: jobdirs.append(Path(out_dir).resolve().parent.parent)          # <job>/<lang>/cNNNN
    planned = False
    for jd in jobdirs:
        try:
            m = json.load(open(jd / "voice_rollout.json", encoding="utf-8"))
            planned = planned or (isinstance(m, dict) and m.get("voice_plan") == 1)
        except FileNotFoundError:
            pass
        except Exception as e:
            raise RuntimeError(f"voice_rollout.json unreadable ({jd}): {e}")
    g = None
    if smap and os.path.exists(smap):
        try:
            g = json.load(open(smap, encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"speaker map unreadable: {e}")
        planned = planned or ("voice_plan" in g)
    if not planned:
        return                                            # legacy job
    if g is None or "voice_plan" not in g:
        raise RuntimeError("planned job but SPEAKER_MAP is missing or carries no voice_plan (chunk fails, no legacy fallback)")
    from sunwai.planning import voices as VA
    try:
        plan = VA.validate_plan(g, lang)
    except VA.PlanError as e:
        raise RuntimeError(f"voice_plan invalid for this job (chunk fails, no legacy fallback): {e}")
    _PLANNED_VOICES = dict(plan["voices"][lang])
    log.info("voice plan: %s voices=%s degraded=%s", lang, _PLANNED_VOICES, plan.get("distinctness_degraded"))


def voice_require_speakers(segments) -> None:
    """planned job: every speaker id present in the chunk must be in the plan. Called OUTSIDE the per-segment handlers, so an
    unexpected id fails the whole chunk instead of degrading into a keep-original fallback."""
    if _PLANNED_VOICES is None:
        return
    missing = sorted({str(sg.get("speaker")) for sg in segments if str(sg.get("speaker")) not in _PLANNED_VOICES})
    if missing:
        raise RuntimeError(f"speaker id(s) {missing} are not in the voice plan (contract error; chunk fails)")


def pick_voice(lang: str, gender: str, speaker_idx: int) -> str:
    """Return the voice for a speaker.

    Uses the job's voice plan when the job is planned, otherwise the legacy per-language gender mapping.

    Args:
        lang: Target language code.
        gender: ``"male"``, ``"female"`` or ``"unknown"``.
        speaker_idx: Speaker index within the job.
    """
    if _PLANNED_VOICES is not None:
        v = _PLANNED_VOICES.get(str(speaker_idx))
        if not v:
            raise RuntimeError(f"voice plan has no voice for speaker {speaker_idx} ({lang})")
        return v
    return _pick_voice_legacy(lang, gender, speaker_idx)


def _pick_voice_legacy(lang: str, gender: str, speaker_idx: int) -> str:
    pools = _voice_pools(lang)
    if pools:
        g = gender if gender in ("male", "female") else ("male" if speaker_idx % 2 == 0 else "female")
        pool = pools.get(g) or pools.get("male") or pools.get("female")
        if pool:
            return pool[(speaker_idx if gender in ("male", "female") else speaker_idx // 2) % len(pool)]
    key = (lang, gender)
    if key in VOICE_MAP:
        return VOICE_MAP[key]
    if gender in DEFAULT_VOICE:
        return DEFAULT_VOICE[gender]
    # unknown gender -> parity alternation for differentiation
    return "Amit" if (speaker_idx % 2 == 0) else "Kavya"
