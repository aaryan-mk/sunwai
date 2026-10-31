"""Reversible per-line expression: candidate revisions that are previewed, applied and undone with full rollback.

Entry points (they return plain dicts and raise :class:`ExprError` on 409/422/...):

* :func:`do_expression_resynth` -- synthesise a candidate for A/B preview;
* :func:`apply_candidate` -- commit it (compare-and-swap), remix, record a new revision;
* :func:`undo_revision` -- create a new revision restoring the previous assets;
* :func:`lipsync_complete` -- accept or reject a lip-sync result as ``"accepted"`` or ``"stale"``.

Kill switch: ``EXPRESSION_ENABLED=0`` (parsed robustly) makes the entry points raise ``ExprError(503)``.
"""

from sunwai.control.expression.commit import apply_candidate, undo_revision
from sunwai.control.expression.errors import ExprError, enabled
from sunwai.control.expression.lipsync import can_undo, lipsync_complete, lipsync_state, live_state
from sunwai.control.expression.planner import INTENT_STYLE, check_wording, plan_expression
from sunwai.control.expression.preview import do_expression_resynth
from sunwai.control.expression.revisions import head_revision
from sunwai.control.expression.voices import resolve_voice

__all__ = [
    "INTENT_STYLE",
    "ExprError",
    "apply_candidate",
    "can_undo",
    "check_wording",
    "do_expression_resynth",
    "enabled",
    "head_revision",
    "lipsync_complete",
    "lipsync_state",
    "live_state",
    "plan_expression",
    "resolve_voice",
    "undo_revision",
]
