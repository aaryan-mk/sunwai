"""Add-language scheduler and source-span (code-switching) records: public facade.

The implementation lives in :mod:`sunwai.jobs.scheduler`; this module keeps the import path that the control
adapters and the server use, and binds the process-wide :class:`LanguageScheduler` instance. It does not import the
server: server seams are injected via :class:`Hooks`. State lives in the job directory (``tracks.json``,
``source_spans.json``, ``source_lang_overrides.json``), always written atomically.
"""

from __future__ import annotations

from sunwai.jobs.scheduler.config import enabled, lead_ms, max_inflight
from sunwai.jobs.scheduler.core import LanguageScheduler
from sunwai.jobs.scheduler.errors import Hooks, LangSchedError
from sunwai.jobs.scheduler.focus import focus_adapter
from sunwai.jobs.scheduler.ordering import chunk_for_ms, plan_order, runway
from sunwai.jobs.scheduler.readiness import track_readiness
from sunwai.jobs.scheduler.requests import LANG_RE
from sunwai.jobs.scheduler.script_lid import (
    classify_text,
    lang_group,
    mask_foreign_terms,
    route_segment,
    script_profile,
    token_profile,
    unmask_terms,
)
from sunwai.jobs.scheduler.source_spans import (
    build_source_spans,
    clear_segment_lang_override,
    load_source_spans,
    persist_source_spans,
    set_segment_lang_override,
)

SCHEDULER = LanguageScheduler()
request_track = SCHEDULER.request_track
release_track = SCHEDULER.release_track
recover_job = SCHEDULER.recover_job
recover_all = SCHEDULER.recover_all
check_voice_plan = SCHEDULER.check_voice_plan

__all__ = [
    "LANG_RE",
    "SCHEDULER",
    "Hooks",
    "LangSchedError",
    "LanguageScheduler",
    "build_source_spans",
    "check_voice_plan",
    "chunk_for_ms",
    "classify_text",
    "clear_segment_lang_override",
    "enabled",
    "focus_adapter",
    "lang_group",
    "lead_ms",
    "load_source_spans",
    "mask_foreign_terms",
    "max_inflight",
    "persist_source_spans",
    "plan_order",
    "recover_all",
    "recover_job",
    "release_track",
    "request_track",
    "route_segment",
    "runway",
    "script_profile",
    "set_segment_lang_override",
    "token_profile",
    "track_readiness",
    "unmask_terms",
]
