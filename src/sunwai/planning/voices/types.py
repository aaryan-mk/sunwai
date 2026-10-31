"""Shared types of the voice planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Bank = dict[str, Any]  # voice_bank.json: voices, langs (per-language voice stats), native, tau
Gate = dict[str, dict[str, float]]  # language -> {"native_cer", "limit"}
Speaker = dict[str, Any]  # {id, gender, p_female, total_s}
Features = dict[str, Any]  # {emb, logf0, evidence_s, clean}


class PlanError(RuntimeError):
    """The voice plan cannot be built (capacity failure) or a stored plan fails validation."""


@dataclass
class VoiceAssignment:
    """Outcome of assigning voices to a job's speakers.

    Attributes:
        canonical: Speaker id -> the voice preferred across languages.
        per_language: Language -> speaker id -> voice actually used there.
        degraded: True when some pair of voices is closer than the distinctness threshold.
        warnings: Human-readable notes (fallbacks, reassignments, degradations).
        detail: Speaker id -> features and similarity recorded in the plan.
        unavailable: Language -> reason no gated assignment exists (only for languages not requested).
    """

    canonical: dict[str, str]
    per_language: dict[str, dict[str, str]]
    degraded: bool
    warnings: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)
