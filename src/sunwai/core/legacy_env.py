"""Mapping from the original flat environment variables to the typed settings groups.

The engine modules still read ``os.environ`` directly. This module keeps both worlds consistent:

* :class:`LegacySource` lets :class:`sunwai.core.config.Settings` read the flat variables (lowest precedence).
* :func:`export_to_environ` writes the typed settings back to the flat variables the engine reads, so a canonical
  ``SUNWAI_<GROUP>__<FIELD>`` value takes effect without touching engine code.

Parsing rules: booleans accept 1/true/yes/on and 0/false/no/off; an unparseable value is ignored here and left in
the environment untouched, so engine behaviour is never changed by this layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class LegacyVar:
    """One flat environment variable and the settings field it feeds.

    Attributes:
        name: Original variable name, e.g. ``PIPE_RUNAWAY_LADDER``.
        group: Settings group attribute, e.g. ``tts``.
        field: Field name inside the group, e.g. ``runaway_ladder``.
        kind: One of ``str``, ``int``, ``float``, ``bool``, ``secret``.
    """

    name: str
    group: str
    field: str
    kind: str

    @property
    def canonical(self) -> str:
        """Return the canonical variable name, ``SUNWAI_<GROUP>__<FIELD>``."""
        return f"SUNWAI_{self.group.upper()}__{self.field.upper()}"

    def parse(self, raw: str) -> Any:
        """Parse ``raw`` into this variable's type; return ``None`` when it cannot be parsed."""
        text = raw.strip()
        try:
            if self.kind == "bool":
                low = text.lower()
                return True if low in _TRUE else False if low in _FALSE else None
            if self.kind == "int":
                return int(text)
            if self.kind == "float":
                return float(text)
        except ValueError:
            return None
        return text if text else None


LEGACY_VARS: tuple[LegacyVar, ...] = (
    LegacyVar("BODHAN_MODE", "bodhan", "mode", "str"),
    LegacyVar("BODHAN_API_BASE", "bodhan", "api_base", "str"),
    LegacyVar("BODHAN_API_MT_FANOUT", "bodhan", "api_mt_fanout", "int"),
    LegacyVar("BODHAN_API_TTS_FANOUT", "bodhan", "api_tts_fanout", "int"),
    LegacyVar("ASR_BASE", "bodhan", "asr_base", "str"),
    LegacyVar("MT_BASE", "bodhan", "mt_base", "str"),
    LegacyVar("MT_BASES", "bodhan", "mt_bases", "str"),
    LegacyVar("TTS_BASE", "bodhan", "tts_base", "str"),
    LegacyVar("IT2_BASE", "bodhan", "it2_base", "str"),
    LegacyVar("IT2_TARGETS", "bodhan", "it2_targets", "str"),
    LegacyVar("SEP_BASE", "bodhan", "sep_base", "str"),
    LegacyVar("DEP_HOST", "bodhan", "dep_host", "str"),
    LegacyVar("TTS_MODEL", "bodhan", "tts_model", "str"),
    LegacyVar("BODHAN_API_KEY", "bodhan", "api_key", "secret"),
    LegacyVar("LLM_PROVIDER", "llm", "provider", "str"),
    LegacyVar("GEMINI_MODEL", "llm", "gemini_model", "str"),
    LegacyVar("DUB_CHAT_MODEL", "llm", "chat_model", "str"),
    LegacyVar("LLM_USAGE_LOG", "llm", "usage_log", "str"),
    LegacyVar("OPENROUTER_REFERER", "llm", "referer", "str"),
    LegacyVar("OPENROUTER_TITLE", "llm", "title", "str"),
    LegacyVar("DUB_CHAT_PRICE_IN", "llm", "chat_price_in", "float"),
    LegacyVar("DUB_CHAT_PRICE_OUT", "llm", "chat_price_out", "float"),
    LegacyVar("OPENROUTER_API_KEY", "llm", "openrouter_api_key", "secret"),
    LegacyVar("GEMINI_API_KEY", "llm", "gemini_api_key", "secret"),
    LegacyVar("FIRST_CHUNK_S", "planning", "first_chunk_s", "float"),
    LegacyVar("CHUNK_LOOKAHEAD", "planning", "chunk_lookahead", "int"),
    LegacyVar("CHUNK_RAMP", "planning", "chunk_ramp", "str"),
    LegacyVar("CHUNK_EXACT_OWN", "planning", "chunk_exact_own", "bool"),
    LegacyVar("CHUNK_GATE_MAX_WAIT", "planning", "chunk_gate_max_wait", "float"),
    LegacyVar("CHUNK_START", "planning", "chunk_start", "float"),
    LegacyVar("CROP_HALFGAP", "planning", "crop_halfgap", "float"),
    LegacyVar("GENDER_KEYFIX", "planning", "gender_keyfix", "bool"),
    LegacyVar("LID_LOCK", "planning", "lid_lock", "bool"),
    LegacyVar("PLANNER_INPUT_SEEK", "planning", "planner_input_seek", "bool"),
    LegacyVar("SPLIT_SPK", "planning", "split_spk", "bool"),
    LegacyVar("SPLIT_MIN_SPK_S", "planning", "split_min_spk_s", "float"),
    LegacyVar("SPLIT_MIN_PIECE_S", "planning", "split_min_piece_s", "float"),
    LegacyVar("PREP_SHARE", "planning", "prep_share", "bool"),
    LegacyVar("YTDLP_COOKIES", "planning", "ytdlp_cookies", "str"),
    LegacyVar("YTDLP_COOKIES_TMPDIR", "planning", "ytdlp_cookies_tmpdir", "str"),
    LegacyVar("SPEAKER_MAP", "planning", "speaker_map", "str"),
    LegacyVar("HF_TOKEN", "planning", "hf_token", "secret"),
    LegacyVar("ASR_WORKERS", "asr", "workers", "int"),
    LegacyVar("ASR_DEDUP", "asr", "dedup", "bool"),
    LegacyVar("ASR_LONG_SPLIT", "asr", "long_split", "bool"),
    LegacyVar("MT_WORKERS", "mt", "workers", "int"),
    LegacyVar("FIT_TRIM", "fit", "trim", "bool"),
    LegacyVar("FIT_MEANING_GUARD", "fit", "meaning_guard", "bool"),
    LegacyVar("FIT_KEEP_LIST", "fit", "keep_list", "bool"),
    LegacyVar("FIT_VOICE_RATE", "fit", "voice_rate", "bool"),
    LegacyVar("REFIT", "fit", "refit", "bool"),
    LegacyVar("REFIT_MIN_CHUNK", "fit", "refit_min_chunk", "int"),
    LegacyVar("SCRIPT_MIN_SHARE", "fit", "script_min_share", "float"),
    LegacyVar("TTS_WORKERS", "tts", "workers", "int"),
    LegacyVar("TTS_NORMALIZE", "tts", "normalize", "bool"),
    LegacyVar("TTS_NORMALIZE_NUMBERS", "tts", "normalize_numbers", "bool"),
    LegacyVar("TTS_SENT_SPLIT", "tts", "sent_split", "bool"),
    LegacyVar("TTS_SENT_WORDSPLIT", "tts", "sent_wordsplit", "bool"),
    LegacyVar("TTS_SENT_DEADLINE_S", "tts", "sent_deadline_s", "float"),
    LegacyVar("TTS_SENT_MAX_CONC", "tts", "sent_max_conc", "int"),
    LegacyVar("TTS_SENT_MAX_PIECES", "tts", "sent_max_pieces", "int"),
    LegacyVar("TTS_SENT_MIN_DRIFT", "tts", "sent_min_drift", "float"),
    LegacyVar("TTS_SENT_MIN_G", "tts", "sent_min_g", "int"),
    LegacyVar("TTS_SENT_MIN_GAP", "tts", "sent_min_gap", "float"),
    LegacyVar("TTS_SENT_SLOW_RTF", "tts", "sent_slow_rtf", "float"),
    LegacyVar("TTS_SLACK_FRAC", "tts", "slack_frac", "float"),
    LegacyVar("TTS_SLACK_MAX_PAUSE", "tts", "slack_max_pause", "float"),
    LegacyVar("TTS_SLACK_PAUSES", "tts", "slack_pauses", "bool"),
    LegacyVar("TTS_SLOW", "tts", "slow", "bool"),
    LegacyVar("TTS_SLOW_MIN", "tts", "slow_min", "float"),
    LegacyVar("DUB_TTS_RETRIES", "tts", "retries", "int"),
    LegacyVar("PIPE_RUNAWAY_GPS", "tts", "runaway_gps", "float"),
    LegacyVar("PIPE_RUNAWAY_LADDER", "tts", "runaway_ladder", "bool"),
    LegacyVar("PIPE_RUNAWAY_LADDER_S", "tts", "runaway_ladder_s", "str"),
    LegacyVar("PIPE_RUNAWAY_MAX_S", "tts", "runaway_max_s", "float"),
    LegacyVar("PIPE_RUNAWAY_TRIES", "tts", "runaway_tries", "int"),
    LegacyVar("MIX_TRIM", "mixing", "trim", "bool"),
    LegacyVar("MIX_TRIM_REL", "mixing", "trim_rel", "float"),
    LegacyVar("MIX_ALIGN_PAUSES", "mixing", "align_pauses", "bool"),
    LegacyVar("MIX_ALIGN_MIN_DRIFT", "mixing", "align_min_drift", "float"),
    LegacyVar("MIX_DUCK", "mixing", "duck", "bool"),
    LegacyVar("MIX_DUCK_DB", "mixing", "duck_db", "float"),
    LegacyVar("MIX_DUCK_ATTACK_MS", "mixing", "duck_attack_ms", "float"),
    LegacyVar("MIX_DUCK_HOLD_MS", "mixing", "duck_hold_ms", "float"),
    LegacyVar("MIX_DUCK_RELEASE_MS", "mixing", "duck_release_ms", "float"),
    LegacyVar("MIX_KEEP_OVERLAP", "mixing", "keep_overlap", "bool"),
    LegacyVar("MIX_OVERLAP_KINDS", "mixing", "overlap_kinds", "str"),
    LegacyVar("MIX_OVERLAP_MAX_CUT", "mixing", "overlap_max_cut", "float"),
    LegacyVar("MIX_KEEP_EVENTS", "mixing", "keep_events", "bool"),
    LegacyVar("MIX_EVENT_THR", "mixing", "event_thr", "float"),
    LegacyVar("MIX_EVENT_MIN_S", "mixing", "event_min_s", "float"),
    LegacyVar("MIX_EVENT_MAX_CAND", "mixing", "event_max_cand", "int"),
    LegacyVar("MIX_KEEP_LEAD", "mixing", "keep_lead", "float"),
    LegacyVar("MIX_KEEP_TAIL", "mixing", "keep_tail", "float"),
    LegacyVar("MIX_LEVEL_FIX", "mixing", "level_fix", "bool"),
    LegacyVar("MIX_LIMITER", "mixing", "limiter", "bool"),
    LegacyVar("MIX_LIM_LOOKAHEAD_MS", "mixing", "lim_lookahead_ms", "float"),
    LegacyVar("MIX_LIM_RELEASE_MS", "mixing", "lim_release_ms", "float"),
    LegacyVar("MIX_DUMP_STEM", "mixing", "dump_stem", "bool"),
    LegacyVar("MIX_AST_DIR", "mixing", "ast_dir", "str"),
    LegacyVar("MUX_FULL_AUDIO", "mixing", "mux_full_audio", "bool"),
    LegacyVar("PIPE_FFMPEG_TIMEOUT", "pipeline", "ffmpeg_timeout", "float"),
    LegacyVar("PIPE_MAX_AUDIO_S", "pipeline", "max_audio_s", "float"),
    LegacyVar("FALLBACK_MAX_FRAC", "pipeline", "fallback_max_frac", "float"),
    LegacyVar("DUB_RETRIES", "pipeline", "retries", "int"),
    LegacyVar("DUB_RETRY_DEADLINE", "pipeline", "retry_deadline", "float"),
    LegacyVar("DUB_SLOTS", "pipeline", "slots", "bool"),
    LegacyVar("DUB_SLOT_DIR", "pipeline", "slot_dir", "str"),
    LegacyVar("DUB_SLOT_WAIT", "pipeline", "slot_wait", "float"),
    LegacyVar("DUB_RESYNTH_TIMEOUT", "pipeline", "resynth_timeout", "float"),
    LegacyVar("DUB_EDIT_GATE_TIMEOUT", "pipeline", "edit_gate_timeout", "float"),
    LegacyVar("EXPRESSION_ENABLED", "pipeline", "expression_enabled", "bool"),
    LegacyVar("EXPR_FFMPEG_TIMEOUT", "pipeline", "expr_ffmpeg_timeout", "float"),
    LegacyVar("LIP_MODE", "lipsync", "mode", "str"),
    LegacyVar("LIP_WORKERS", "lipsync", "workers", "str"),
    LegacyVar("LIP_ASD", "lipsync", "asd", "bool"),
    LegacyVar("LIP_ASD_BS", "lipsync", "asd_bs", "int"),
    LegacyVar("LIP_ASD_CONF", "lipsync", "asd_conf", "float"),
    LegacyVar("LIP_ASD_OFFMAX", "lipsync", "asd_offmax", "int"),
    LegacyVar("LIP_ASD_STATIC", "lipsync", "asd_static", "float"),
    LegacyVar("LIP_ASD_STATIC_SYNC", "lipsync", "asd_static_sync", "bool"),
    LegacyVar("LIP_EMPTY_CACHE", "lipsync", "empty_cache", "bool"),
    LegacyVar("LIP_GPU_FRAC", "lipsync", "gpu_frac", "float"),
    LegacyVar("LIP_MULTIFACE", "lipsync", "multiface", "bool"),
    LegacyVar("LIP_PASTE_V2", "lipsync", "paste_v2", "bool"),
    LegacyVar("LATENTSYNC_DIR", "lipsync", "latentsync_dir", "str"),
    LegacyVar("DUB_ACTIONS", "control", "actions", "bool"),
    LegacyVar("DUB_CTL_TOKEN", "control", "token", "secret"),
    LegacyVar("DUB_CHAT", "control", "chat", "bool"),
    LegacyVar("DUB_ADD_LANG", "control", "add_lang", "bool"),
    LegacyVar("DUB_CTL_ADDLANG", "control", "addlang", "bool"),
    LegacyVar("DUB_CTL_ADDLANG_S", "control", "addlang_s", "float"),
    LegacyVar("DUB_CTL_CONFIRM_TTL", "control", "confirm_ttl", "float"),
    LegacyVar("DUB_CTL_DAILY_SYNTH_S", "control", "daily_synth_s", "float"),
    LegacyVar("DUB_CTL_DAILY_USD", "control", "daily_usd", "float"),
    LegacyVar("DUB_CTL_DEVHOOKS", "control", "devhooks", "bool"),
    LegacyVar("DUB_CTL_DRYRUN", "control", "dryrun", "bool"),
    LegacyVar("DUB_CTL_DRYRUN_DELAY", "control", "dryrun_delay", "float"),
    LegacyVar("DUB_CTL_EV_KEEP", "control", "ev_keep", "int"),
    LegacyVar("DUB_CTL_EV_MAX_BYTES", "control", "ev_max_bytes", "int"),
    LegacyVar("DUB_CTL_EV_REPLAY", "control", "ev_replay", "int"),
    LegacyVar("DUB_CTL_GEMINI_PER_MIN", "control", "gemini_per_min", "int"),
    LegacyVar("DUB_CTL_GLOBAL_SYNTH", "control", "global_synth", "int"),
    LegacyVar("DUB_CTL_IP_ACTIONS_PER_MIN", "control", "ip_actions_per_min", "int"),
    LegacyVar("DUB_CTL_IP_CHAT_PER_MIN", "control", "ip_chat_per_min", "int"),
    LegacyVar("DUB_CTL_IP_POLL_PER_MIN", "control", "ip_poll_per_min", "int"),
    LegacyVar("DUB_CTL_IP_READ_PER_MIN", "control", "ip_read_per_min", "int"),
    LegacyVar("DUB_CTL_LANG_MAX", "control", "lang_max", "int"),
    LegacyVar("DUB_CTL_LANG_MAX_ERRS", "control", "lang_max_errs", "int"),
    LegacyVar("DUB_CTL_LANG_MAX_ERR_S", "control", "lang_max_err_s", "float"),
    LegacyVar("DUB_CTL_LANG_POLL", "control", "lang_poll", "float"),
    LegacyVar("DUB_CTL_LOCK_WAIT_S", "control", "lock_wait_s", "float"),
    LegacyVar("DUB_CTL_LS_POLL", "control", "ls_poll", "float"),
    LegacyVar("DUB_CTL_LS_WAIT", "control", "ls_wait", "float"),
    LegacyVar("DUB_CTL_M2_TIMEOUT_S", "control", "m2_timeout_s", "float"),
    LegacyVar("DUB_CTL_MAX_SYNTH", "control", "max_synth", "int"),
    LegacyVar("DUB_CTL_PREVIEWS_PER_MIN", "control", "previews_per_min", "int"),
    LegacyVar("DUB_CTL_SPEND_CAP_USD", "control", "spend_cap_usd", "float"),
    LegacyVar("DUB_CTL_SSE_PER_JOB", "control", "sse_per_job", "int"),
    LegacyVar("DUB_CTL_SSE_TOTAL", "control", "sse_total", "int"),
    LegacyVar("DUB_CTL_SYNTH_CAP_S", "control", "synth_cap_s", "float"),
    LegacyVar("DUB_ROOT", "server", "dub_root", "str"),
    LegacyVar("PORT", "server", "port", "int"),
    LegacyVar("MAX_CONCURRENT", "server", "max_concurrent", "int"),
    LegacyVar("DUB_TOKEN", "server", "token", "secret"),
    LegacyVar("DUB_CORS_ORIGINS", "server", "cors_origins", "str"),
    LegacyVar("DUB_ADMISSION", "server", "admission", "bool"),
    LegacyVar("DUB_FANOUT", "server", "fanout", "int"),
    LegacyVar("DUB_FAULT", "server", "fault", "str"),
    LegacyVar("DUB_MAX_DUR_S", "server", "max_dur_s", "float"),
    LegacyVar("DUB_MAX_EDIT_CHARS", "server", "max_edit_chars", "int"),
    LegacyVar("DUB_MAX_QUEUED_JOBS", "server", "max_queued_jobs", "int"),
    LegacyVar("DUB_MAX_START_S", "server", "max_start_s", "float"),
    LegacyVar("DUB_MAX_TARGETS", "server", "max_targets", "int"),
    LegacyVar("DUB_MAX_UNCACHED_JOBS", "server", "max_uncached_jobs", "int"),
    LegacyVar("DUB_MAX_VIDEO_S", "server", "max_video_s", "float"),
    LegacyVar("DUB_MAX_WORK_S", "server", "max_work_s", "float"),
    LegacyVar("DUB_RECONCILE", "server", "reconcile", "bool"),
    LegacyVar("DUB_SERVICES", "server", "services", "str"),
    LegacyVar("IT2_MAX_BODY", "mirrors", "it2_max_body", "int"),
    LegacyVar("IT2_MAX_CHARS", "mirrors", "it2_max_chars", "int"),
    LegacyVar("IT2_MAX_PENDING", "mirrors", "it2_max_pending", "int"),
    LegacyVar("IT2_MAX_TOKENS", "mirrors", "it2_max_tokens", "int"),
    LegacyVar("IT2_MODELS_DIR", "mirrors", "it2_models_dir", "str"),
    LegacyVar("SEP_ROOT", "mirrors", "sep_root", "str"),
    LegacyVar("SEP_GPU", "mirrors", "sep_gpu", "str"),
    LegacyVar("SEP_DECODE_PARALLEL", "mirrors", "sep_decode_parallel", "int"),
    LegacyVar("SEP_MAX_BYTES", "mirrors", "sep_max_bytes", "int"),
    LegacyVar("SEP_MAX_PENDING", "mirrors", "sep_max_pending", "int"),
    LegacyVar("SEP_QUEUE_TIMEOUT", "mirrors", "sep_queue_timeout", "float"),
)

BY_NAME: dict[str, LegacyVar] = {v.name: v for v in LEGACY_VARS}


def encode(value: Any) -> str:
    """Render a settings value the way the engine expects to read it from the environment."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if hasattr(value, "get_secret_value"):
        return str(value.get_secret_value())
    return str(value)


class LegacySource(PydanticBaseSettingsSource):
    """Settings source that reads the original flat environment variables (lowest-priority env source)."""

    def __init__(self, settings_cls: type[BaseSettings], environ: Mapping[str, str] | None = None) -> None:
        super().__init__(settings_cls)
        self._environ = os.environ if environ is None else environ

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        """Unused: values are produced in bulk by :meth:`__call__`."""
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        """Return ``{group: {field: value}}`` for every legacy variable present and parseable."""
        data: dict[str, dict[str, Any]] = {}
        for var in LEGACY_VARS:
            raw = self._environ.get(var.name)
            if raw is None:
                continue
            value = var.parse(raw)
            if value is not None:
                data.setdefault(var.group, {})[var.field] = value
        return data


def export_to_environ(settings: Any, environ: MutableMapping[str, str] | None = None) -> list[str]:
    """Write typed settings to the flat variables the engine reads.

    A canonical ``SUNWAI_<GROUP>__<FIELD>`` variable overrides an existing flat variable; otherwise an existing flat
    variable is left exactly as it is. Unset (``None``) fields are never written.

    Args:
        settings: A :class:`sunwai.core.config.Settings` instance.
        environ: Target mapping, ``os.environ`` by default.

    Returns:
        Names of the flat variables that were written.
    """
    env = os.environ if environ is None else environ
    written: list[str] = []
    for var in LEGACY_VARS:
        value = getattr(getattr(settings, var.group), var.field)
        if value is None:
            continue
        if var.canonical in env or var.name not in env:
            env[var.name] = encode(value)
            written.append(var.name)
    return written
