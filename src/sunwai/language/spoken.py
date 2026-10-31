"""Which text is sent to TTS for a segment: the expression text, the number-normalised text, or the translation."""

from __future__ import annotations

from typing import Any, Dict

from sunwai.language.numbers import _TN_VER, _tn_flag
from sunwai.pipeline.support.expression_flags import _expr_on

_NUM_STAGE_LANG: str | None = None
"""Language of the last ``stage_numbers`` run in this process; a stored ``num_text`` is valid only for it."""


def set_stage_language(lang: str | None) -> None:
    """Record the target language of the current ``stage_numbers`` run."""
    global _NUM_STAGE_LANG
    _NUM_STAGE_LANG = lang


def _num_spoken(seg: Dict[str, Any]) -> str:
    """tgt_text, or its number-normalised spoken form while ALL of these hold: it was made from the current tgt_text by this algorithm
    version and this process's target language, the switches are on and the segment is not edited / failed / fallback / expression."""
    t = seg.get("tgt_text") or ""
    if (seg.get("num_text") and seg.get("num_src") == t and seg.get("num_ver") == _TN_VER and seg.get("num_lang") == _NUM_STAGE_LANG
            and _NUM_STAGE_LANG is not None and not seg.get("edited") and not seg.get("error") and not seg.get("fallback")
            and not _expr_on(seg) and _tn_flag("TTS_NORMALIZE", True) and _tn_flag("TTS_NORMALIZE_NUMBERS", True)):
        return seg["num_text"]
    return t


def _expr_spoken(seg):
    """text sent to TTS: the m2 tts_text only while _expr_on(seg); display text (tgt_text, subtitles) is never changed."""
    return seg["tts_text"] if _expr_on(seg) else _num_spoken(seg)      # v3: number words only for clear quantities
