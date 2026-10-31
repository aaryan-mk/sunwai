"""Expression (delivery control) switches and style helpers."""

from __future__ import annotations

import logging
import os


log = logging.getLogger(__name__)



def _expr_style(seg):
    """a style reaches TTS/fingerprints ONLY when the expression feature is ON and the style was set by the expression module
    (style_src == "m2"). Any other segment 'style' key (pre-existing data) is ignored: OFF is inert."""
    if _expr_on(seg) and seg.get("style_src") == "m2":
        st = seg.get("style")
        return st if isinstance(st, str) and st.strip() else None
    return None


_EXPR_FLAG = None


def _expr_flag():
    """ON only if EXPRESSION_ENABLED is truthy (default OFF).
    Evaluated once per process (each chunk runs in its own subprocess); any filesystem error counts as OFF."""
    global _EXPR_FLAG
    if _EXPR_FLAG is None:
        v = (os.environ.get("EXPRESSION_ENABLED") or "").strip().lower()
        if v in ("0", "false", "no", "off"):
            _EXPR_FLAG = False
        elif v in ("1", "true", "yes", "on"):
            _EXPR_FLAG = True
        else:
            _EXPR_FLAG = False                      # default OFF
    return _EXPR_FLAG


def _expr_on(seg):
    """expression feature ON, segment carries an m2 expression, and it was made for the CURRENT display text (a later text edit voids it)."""
    return (seg.get("tts_by") == "m2" and bool(seg.get("tts_text")) and seg.get("tts_src") == seg.get("tgt_text")
            and _expr_flag())
