"""Names of planner products and the rule for invalidating the ones derived from a replaced input."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable

from sunwai.planning.fileio import json_or_none, remove_file

log = logging.getLogger("planner")

#: Products computed from ``source16k.wav``; all become stale when the wav is replaced.
DERIVED_FROM_WAV = (
    "vad.json",
    "plan.json",
    "spkturns.json",
    "spkbackend.json",
    "spkgender.json",
    "speakers_global.json",
)

SPEAKER_MAP = "speakers_global.json"

#: Prefix of temp files/dirs that a crashed run may leave behind (swept by stage 1).
TMP_PREFIX = ".g066tmp_"


def has_voice_plan(out_dir: str) -> bool:
    """True when the job carries a voice plan (rollout marker 1, or a plan already inside the speaker map)."""
    marker = json_or_none(os.path.join(out_dir, "voice_rollout.json"))
    speakers = json_or_none(os.path.join(out_dir, SPEAKER_MAP))
    marked = isinstance(marker, dict) and marker.get("voice_plan") == 1
    embedded = isinstance(speakers, dict) and isinstance(speakers.get("voice_plan"), dict)
    return marked or embedded


def remove_derived(out_dir: str, names: Iterable[str]) -> None:
    """Delete stale derived products.

    The speaker map of a job that carries a voice plan is kept: a voice plan is immutable, so every invalidation
    path keeps it and the later speaker-map write hard-fails on a changed map.
    """
    keep_map = has_voice_plan(out_dir)
    for name in names:
        if keep_map and name == SPEAKER_MAP:
            log.warning("[voice] keeping speakers_global.json (immutable voice plan) through invalidation of %s", name)
            continue
        remove_file(os.path.join(out_dir, name))
