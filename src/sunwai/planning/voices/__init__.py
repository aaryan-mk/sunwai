"""Voice assignment: one distinct, intelligibility-gated bank voice per speaker and language.

The planner computes a ``voice_plan`` ONCE per new job and embeds it in ``speakers_global.json``; workers only
validate and read it. The assignment function is pluggable: :func:`plan_job` takes it as a parameter.
"""

from sunwai.planning.voices.assignment import VoicePlanner
from sunwai.planning.voices.catalogue import PLAN_VERSION, catalogue_rev, load_bank, smap_hash
from sunwai.planning.voices.features import speaker_features
from sunwai.planning.voices.gate import cer_gate, eligible
from sunwai.planning.voices.plan import assign_voices, plan_job, validate_plan
from sunwai.planning.voices.types import PlanError, VoiceAssignment

__all__ = [
    "PLAN_VERSION",
    "PlanError",
    "VoiceAssignment",
    "VoicePlanner",
    "assign_voices",
    "catalogue_rev",
    "cer_gate",
    "eligible",
    "load_bank",
    "plan_job",
    "smap_hash",
    "speaker_features",
    "validate_plan",
]
