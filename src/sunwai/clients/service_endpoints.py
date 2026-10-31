"""Resolved service endpoints, read once at import from ``BODHAN_MODE`` and the per-service overrides.

Tests that change the environment reload this module (and then ``service_calls``) to pick the new values up.
"""

from __future__ import annotations

import itertools
import os
import random

from sunwai.clients import bodhan_endpoints as bodhan_cfg

BASE_URL = bodhan_cfg.api_base()

# Service endpoints come from one place (BODHAN_MODE api|local, see sunwai.clients.bodhan_endpoints).
ASR_BASE = bodhan_cfg.asr_base()

MT_BASES = bodhan_cfg.mt_bases()

MT_BASE = MT_BASES[0] if MT_BASES else ""

TTS_BASES = bodhan_cfg.tts_bases()

TTS_BASE = ",".join(TTS_BASES)

IT2_BASE = bodhan_cfg.it2_base()

IT2_TARGETS = set(os.environ.get("IT2_TARGETS", "mni,sd").split(","))

_TTS_RR = itertools.count()

def _next_tts_base():
    """Return the next TTS replica base URL (round robin)."""
    if not TTS_BASES:
        raise RuntimeError("no TTS endpoint configured")
    return TTS_BASES[next(_TTS_RR) % len(TTS_BASES)]

def _next_mt_base():
    """Return a random MT replica base URL (spreads load across serial replicas)."""
    if not MT_BASES:
        raise RuntimeError("no MT endpoint configured")
    return random.choice(MT_BASES)

TTS_MODEL = os.environ.get("TTS_MODEL", "indic-speak")

