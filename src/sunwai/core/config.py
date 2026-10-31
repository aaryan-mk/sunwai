"""Typed SunwAI settings, grouped by domain. GENERATED from the legacy-variable table; edit ``legacy_env`` and regenerate.

Every field defaults to ``None``, which means "use the engine's built-in default". Canonical environment variables are
``SUNWAI_<GROUP>__<FIELD>`` (for example ``SUNWAI_TTS__RUNAWAY_LADDER=false``). Each field also accepts the original
flat variable (for example ``PIPE_RUNAWAY_LADDER``); the mapping lives in :mod:`sunwai.core.legacy_env`. Precedence:
explicit init values, canonical environment, ``.env``, legacy variables, defaults.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from sunwai.core.legacy_env import LegacySource


class ConfigGroup(BaseModel):
    """Base class for one domain group: immutable, unknown keys rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class BodhanSettings(ConfigGroup):
    """Bodhan ASR/MT/TTS endpoints and credentials."""

    mode: str | None = Field(
        default=None,
        description='Where ASR/MT/TTS requests go: "api" (hosted, default) or "local" (self-hosted replicas). Legacy name: BODHAN_MODE.',
    )
    api_base: str | None = Field(default=None, description="Hosted Bodhan API base URL. Legacy name: BODHAN_API_BASE.")
    api_mt_fanout: int | None = Field(
        default=None,
        description="Logical MT replica count in api mode (sizes worker pools). Legacy name: BODHAN_API_MT_FANOUT.",
    )
    api_tts_fanout: int | None = Field(
        default=None,
        description="Logical TTS replica count in api mode (sizes worker pools). Legacy name: BODHAN_API_TTS_FANOUT.",
    )
    asr_base: str | None = Field(default=None, description="Local-mode ASR base URL. Legacy name: ASR_BASE.")
    mt_base: str | None = Field(default=None, description="Local-mode single MT base URL. Legacy name: MT_BASE.")
    mt_bases: str | None = Field(
        default=None, description="Local-mode comma-separated MT replica URLs. Legacy name: MT_BASES."
    )
    tts_base: str | None = Field(
        default=None, description="Local-mode comma-separated TTS replica URLs. Legacy name: TTS_BASE."
    )
    it2_base: str | None = Field(
        default=None, description="IndicTrans2 mirror base URL (always local). Legacy name: IT2_BASE."
    )
    it2_targets: str | None = Field(
        default=None,
        description="Comma-separated target languages routed to IndicTrans2 instead of Bodhan MT. Legacy name: IT2_TARGETS.",
    )
    sep_base: str | None = Field(
        default=None, description="Source-separation service base URL (always local). Legacy name: SEP_BASE."
    )
    dep_host: str | None = Field(
        default=None,
        description="Host used for local-mode default URLs and the supervisor probes. Legacy name: DEP_HOST.",
    )
    tts_model: str | None = Field(
        default=None, description="TTS model id sent to the synthesis endpoint. Legacy name: TTS_MODEL."
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="Shared Bodhan API key (per-service BODHAN_<ASR|MT|TTS>_KEY override it). Legacy name: BODHAN_API_KEY.",
    )


class LlmSettings(ConfigGroup):
    """LLM provider and models."""

    provider: str | None = Field(
        default=None, description='LLM backend: "openrouter" (default) or "gemini". Legacy name: LLM_PROVIDER.'
    )
    gemini_model: str | None = Field(
        default=None,
        description="Default model name (mapped to google/<name> on OpenRouter). Legacy name: GEMINI_MODEL.",
    )
    chat_model: str | None = Field(
        default=None, description="Model for the control-plane chat planner. Legacy name: DUB_CHAT_MODEL."
    )
    usage_log: str | None = Field(
        default=None,
        description="Append-only JSONL path for per-call token and cost records. Legacy name: LLM_USAGE_LOG.",
    )
    referer: str | None = Field(
        default=None, description="HTTP-Referer header sent to OpenRouter. Legacy name: OPENROUTER_REFERER."
    )
    title: str | None = Field(
        default=None, description="X-Title header sent to OpenRouter. Legacy name: OPENROUTER_TITLE."
    )
    chat_price_in: float | None = Field(
        default=None,
        description="Assumed USD per million input tokens for chat cost accounting. Legacy name: DUB_CHAT_PRICE_IN.",
    )
    chat_price_out: float | None = Field(
        default=None,
        description="Assumed USD per million output tokens for chat cost accounting. Legacy name: DUB_CHAT_PRICE_OUT.",
    )
    openrouter_api_key: SecretStr | None = Field(
        default=None, description="OpenRouter API key. Legacy name: OPENROUTER_API_KEY."
    )
    gemini_api_key: SecretStr | None = Field(
        default=None,
        description="Google Generative Language API key (only when LLM_PROVIDER=gemini). Legacy name: GEMINI_API_KEY.",
    )


class PlanningSettings(ConfigGroup):
    """Source preparation, chunking and speaker planning."""

    first_chunk_s: float | None = Field(
        default=None,
        description="Target length of the first (latency-critical) chunk, seconds. Legacy name: FIRST_CHUNK_S.",
    )
    chunk_lookahead: int | None = Field(
        default=None, description="How many chunks a language may run ahead of playback. Legacy name: CHUNK_LOOKAHEAD."
    )
    chunk_ramp: str | None = Field(
        default=None, description="Chunk-length ramp used to shorten early chunks. Legacy name: CHUNK_RAMP."
    )
    chunk_exact_own: bool | None = Field(
        default=None, description="Cut each chunk on exact frame ownership boundaries. Legacy name: CHUNK_EXACT_OWN."
    )
    chunk_gate_max_wait: float | None = Field(
        default=None,
        description="Maximum seconds a chunk waits for its predecessor gate. Legacy name: CHUNK_GATE_MAX_WAIT.",
    )
    chunk_start: float | None = Field(
        default=None,
        description="Absolute chunk start offset passed by the server to chunk processes. Legacy name: CHUNK_START.",
    )
    crop_halfgap: float | None = Field(
        default=None, description="Half-gap in seconds kept around speaker crops. Legacy name: CROP_HALFGAP."
    )
    gender_keyfix: bool | None = Field(
        default=None, description="Enable the gender cache-key fix. Legacy name: GENDER_KEYFIX."
    )
    lid_lock: bool | None = Field(
        default=None, description="Serialise source-language identification across processes. Legacy name: LID_LOCK."
    )
    planner_input_seek: bool | None = Field(
        default=None,
        description="Seek in the input when downloading a window instead of cutting afterwards. Legacy name: PLANNER_INPUT_SEEK.",
    )
    split_spk: bool | None = Field(default=None, description="Split slots at speaker changes. Legacy name: SPLIT_SPK.")
    split_min_spk_s: float | None = Field(
        default=None, description="Minimum speaker turn length for a split, seconds. Legacy name: SPLIT_MIN_SPK_S."
    )
    split_min_piece_s: float | None = Field(
        default=None, description="Minimum piece length after a split, seconds. Legacy name: SPLIT_MIN_PIECE_S."
    )
    prep_share: bool | None = Field(
        default=None,
        description="Share the language-independent front half across target languages. Legacy name: PREP_SHARE.",
    )
    ytdlp_cookies: str | None = Field(
        default=None, description="Path to a mode-600 cookies file for yt-dlp (optional). Legacy name: YTDLP_COOKIES."
    )
    ytdlp_cookies_tmpdir: str | None = Field(
        default=None, description="Private directory for per-download cookie copies. Legacy name: YTDLP_COOKIES_TMPDIR."
    )
    speaker_map: str | None = Field(
        default=None,
        description="Path to speakers_global.json, set by the server for chunk processes. Legacy name: SPEAKER_MAP.",
    )
    hf_token: SecretStr | None = Field(
        default=None, description="Hugging Face token for gated pyannote weights. Legacy name: HF_TOKEN."
    )


class AsrSettings(ConfigGroup):
    """ASR stage tuning."""

    workers: int | None = Field(
        default=None, description="Concurrent ASR calls per chunk process. Legacy name: ASR_WORKERS."
    )
    dedup: bool | None = Field(
        default=None, description="Deduplicate words repeated across slot boundaries. Legacy name: ASR_DEDUP."
    )
    long_split: bool | None = Field(
        default=None, description="Split over-long slots before transcription. Legacy name: ASR_LONG_SPLIT."
    )


class MtSettings(ConfigGroup):
    """MT stage tuning."""

    workers: int | None = Field(
        default=None, description="Concurrent MT calls per chunk process. Legacy name: MT_WORKERS."
    )


class FitSettings(ConfigGroup):
    """Duration-fit stage."""

    trim: bool | None = Field(
        default=None, description="Allow the fit stage to trim text to the duration budget. Legacy name: FIT_TRIM."
    )
    meaning_guard: bool | None = Field(
        default=None,
        description="Reject fit rewrites that change numbers, names or negations. Legacy name: FIT_MEANING_GUARD.",
    )
    keep_list: bool | None = Field(
        default=None, description="Protect listed terms from rewriting. Legacy name: FIT_KEEP_LIST."
    )
    voice_rate: bool | None = Field(
        default=None,
        description="Budget each line with the assigned voice's measured speaking rate. Legacy name: FIT_VOICE_RATE.",
    )
    refit: bool | None = Field(
        default=None, description="Re-fit and re-synthesise lines that overrun after TTS. Legacy name: REFIT."
    )
    refit_min_chunk: int | None = Field(
        default=None,
        description="First chunk index allowed to refit (earlier chunks skip it for latency). Legacy name: REFIT_MIN_CHUNK.",
    )
    script_min_share: float | None = Field(
        default=None,
        description="Minimum share of target-script letters for an accepted line. Legacy name: SCRIPT_MIN_SHARE.",
    )


class TtsSettings(ConfigGroup):
    """TTS stage: splitting, normalisation, pacing and runaway repair."""

    workers: int | None = Field(
        default=None, description="Concurrent TTS calls per chunk process. Legacy name: TTS_WORKERS."
    )
    normalize: bool | None = Field(
        default=None, description="Normalise text for speech before synthesis. Legacy name: TTS_NORMALIZE."
    )
    normalize_numbers: bool | None = Field(
        default=None,
        description="Spell numbers out as words for the target language. Legacy name: TTS_NORMALIZE_NUMBERS.",
    )
    sent_split: bool | None = Field(
        default=None, description="Synthesise long lines sentence by sentence. Legacy name: TTS_SENT_SPLIT."
    )
    sent_wordsplit: bool | None = Field(
        default=None,
        description="Allow word-level splits when no sentence boundary exists. Legacy name: TTS_SENT_WORDSPLIT.",
    )
    sent_deadline_s: float | None = Field(
        default=None, description="Time budget for a split synthesis, seconds. Legacy name: TTS_SENT_DEADLINE_S."
    )
    sent_max_conc: int | None = Field(
        default=None, description="Maximum concurrent pieces per split line. Legacy name: TTS_SENT_MAX_CONC."
    )
    sent_max_pieces: int | None = Field(
        default=None, description="Maximum pieces a line may be split into. Legacy name: TTS_SENT_MAX_PIECES."
    )
    sent_min_drift: float | None = Field(
        default=None, description="Minimum timing drift before a split is attempted. Legacy name: TTS_SENT_MIN_DRIFT."
    )
    sent_min_g: int | None = Field(
        default=None, description="Minimum graphemes for a split piece. Legacy name: TTS_SENT_MIN_G."
    )
    sent_min_gap: float | None = Field(
        default=None, description="Minimum source pause that justifies a split, seconds. Legacy name: TTS_SENT_MIN_GAP."
    )
    sent_slow_rtf: float | None = Field(
        default=None, description="Real-time factor above which a split is skipped. Legacy name: TTS_SENT_SLOW_RTF."
    )
    slack_frac: float | None = Field(
        default=None,
        description="Fraction of slot slack that may be absorbed by stretching pauses. Legacy name: TTS_SLACK_FRAC.",
    )
    slack_max_pause: float | None = Field(
        default=None, description="Maximum pause stretch, seconds. Legacy name: TTS_SLACK_MAX_PAUSE."
    )
    slack_pauses: bool | None = Field(
        default=None, description="Absorb overruns by stretching pauses. Legacy name: TTS_SLACK_PAUSES."
    )
    slow: bool | None = Field(
        default=None, description="Slow slightly-fast lines down toward the slot. Legacy name: TTS_SLOW."
    )
    slow_min: float | None = Field(
        default=None, description="Minimum tempo factor when slowing. Legacy name: TTS_SLOW_MIN."
    )
    retries: int | None = Field(default=None, description="HTTP retries per TTS call. Legacy name: DUB_TTS_RETRIES.")
    runaway_gps: float | None = Field(
        default=None,
        description="Graphemes-per-second ceiling that flags a runaway synthesis. Legacy name: PIPE_RUNAWAY_GPS.",
    )
    runaway_ladder: bool | None = Field(
        default=None, description="Enable the runaway repair ladder. Legacy name: PIPE_RUNAWAY_LADDER."
    )
    runaway_ladder_s: str | None = Field(
        default=None,
        description="Time budget per rung of the runaway repair ladder. Legacy name: PIPE_RUNAWAY_LADDER_S.",
    )
    runaway_max_s: float | None = Field(
        default=None,
        description="Absolute duration that always counts as a runaway, seconds. Legacy name: PIPE_RUNAWAY_MAX_S.",
    )
    runaway_tries: int | None = Field(
        default=None, description="Maximum repair attempts for a runaway line. Legacy name: PIPE_RUNAWAY_TRIES."
    )


class MixingSettings(ConfigGroup):
    """Mix stage: alignment, ducking, preservation and limiting."""

    trim: bool | None = Field(
        default=None, description="Trim leading and trailing silence of synthesised speech. Legacy name: MIX_TRIM."
    )
    trim_rel: float | None = Field(
        default=None, description="Relative threshold for silence trimming. Legacy name: MIX_TRIM_REL."
    )
    align_pauses: bool | None = Field(
        default=None, description="Align dub pauses to the source pauses. Legacy name: MIX_ALIGN_PAUSES."
    )
    align_min_drift: float | None = Field(
        default=None,
        description="Minimum drift before pause alignment acts, seconds. Legacy name: MIX_ALIGN_MIN_DRIFT.",
    )
    duck: bool | None = Field(default=None, description="Duck the background under speech. Legacy name: MIX_DUCK.")
    duck_db: float | None = Field(default=None, description="Ducking depth, dB. Legacy name: MIX_DUCK_DB.")
    duck_attack_ms: float | None = Field(
        default=None, description="Ducking attack time, ms. Legacy name: MIX_DUCK_ATTACK_MS."
    )
    duck_hold_ms: float | None = Field(
        default=None, description="Ducking hold time, ms. Legacy name: MIX_DUCK_HOLD_MS."
    )
    duck_release_ms: float | None = Field(
        default=None, description="Ducking release time, ms. Legacy name: MIX_DUCK_RELEASE_MS."
    )
    keep_overlap: bool | None = Field(
        default=None, description="Preserve overlapping source speech in the mix. Legacy name: MIX_KEEP_OVERLAP."
    )
    overlap_kinds: str | None = Field(
        default=None, description="Which overlap kinds are preserved. Legacy name: MIX_OVERLAP_KINDS."
    )
    overlap_max_cut: float | None = Field(
        default=None, description="Maximum seconds cut when removing overlaps. Legacy name: MIX_OVERLAP_MAX_CUT."
    )
    keep_events: bool | None = Field(
        default=None,
        description="Preserve non-speech events (laughter, applause) from the source. Legacy name: MIX_KEEP_EVENTS.",
    )
    event_thr: float | None = Field(
        default=None, description="Classifier threshold for non-speech events. Legacy name: MIX_EVENT_THR."
    )
    event_min_s: float | None = Field(
        default=None, description="Minimum event length, seconds. Legacy name: MIX_EVENT_MIN_S."
    )
    event_max_cand: int | None = Field(
        default=None, description="Maximum candidate events examined per chunk. Legacy name: MIX_EVENT_MAX_CAND."
    )
    keep_lead: float | None = Field(
        default=None, description="Seconds of source kept before a preserved event. Legacy name: MIX_KEEP_LEAD."
    )
    keep_tail: float | None = Field(
        default=None, description="Seconds of source kept after a preserved event. Legacy name: MIX_KEEP_TAIL."
    )
    level_fix: bool | None = Field(
        default=None, description="Match dub loudness to the source. Legacy name: MIX_LEVEL_FIX."
    )
    limiter: bool | None = Field(default=None, description="Apply the output peak limiter. Legacy name: MIX_LIMITER.")
    lim_lookahead_ms: float | None = Field(
        default=None, description="Limiter look-ahead, ms. Legacy name: MIX_LIM_LOOKAHEAD_MS."
    )
    lim_release_ms: float | None = Field(
        default=None, description="Limiter release, ms. Legacy name: MIX_LIM_RELEASE_MS."
    )
    dump_stem: bool | None = Field(
        default=None, description="Write intermediate stems for debugging. Legacy name: MIX_DUMP_STEM."
    )
    ast_dir: str | None = Field(
        default=None,
        description="Glob of a local AudioSet-AST snapshot used for event detection. Legacy name: MIX_AST_DIR.",
    )
    mux_full_audio: bool | None = Field(
        default=None,
        description="Mux the full chunk audio rather than only the speech window. Legacy name: MUX_FULL_AUDIO.",
    )


class PipelineSettings(ConfigGroup):
    """Pipeline-wide limits, retries and expression."""

    ffmpeg_timeout: float | None = Field(
        default=None, description="Timeout for ffmpeg subprocesses, seconds. Legacy name: PIPE_FFMPEG_TIMEOUT."
    )
    max_audio_s: float | None = Field(
        default=None, description="Longest audio accepted from a service, seconds. Legacy name: PIPE_MAX_AUDIO_S."
    )
    fallback_max_frac: float | None = Field(
        default=None,
        description="Maximum fraction of speech allowed to use degraded fallbacks. Legacy name: FALLBACK_MAX_FRAC.",
    )
    retries: int | None = Field(default=None, description="HTTP retries per service call. Legacy name: DUB_RETRIES.")
    retry_deadline: float | None = Field(
        default=None,
        description="Overall deadline for retrying one service call, seconds. Legacy name: DUB_RETRY_DEADLINE.",
    )
    slots: bool | None = Field(
        default=None, description="Enable cross-process service concurrency slots. Legacy name: DUB_SLOTS."
    )
    slot_dir: str | None = Field(
        default=None, description="Directory holding the cross-process slot lock files. Legacy name: DUB_SLOT_DIR."
    )
    slot_wait: float | None = Field(
        default=None, description="Maximum wait for a service slot, seconds. Legacy name: DUB_SLOT_WAIT."
    )
    resynth_timeout: float | None = Field(
        default=None, description="Timeout for a single-line re-synthesis, seconds. Legacy name: DUB_RESYNTH_TIMEOUT."
    )
    edit_gate_timeout: float | None = Field(
        default=None, description="Timeout waiting for the edit gate, seconds. Legacy name: DUB_EDIT_GATE_TIMEOUT."
    )
    expression_enabled: bool | None = Field(
        default=None, description="Enable expression (delivery) controls. Legacy name: EXPRESSION_ENABLED."
    )
    expr_ffmpeg_timeout: float | None = Field(
        default=None,
        description="Timeout for expression-preview ffmpeg calls, seconds. Legacy name: EXPR_FFMPEG_TIMEOUT.",
    )


class LipsyncSettings(ConfigGroup):
    """Lip-sync worker and gates."""

    mode: str | None = Field(
        default=None, description='Lip-sync dispatch: "proc" (worker processes) or "none". Legacy name: LIP_MODE.'
    )
    workers: str | None = Field(
        default=None, description="Comma-separated lip-sync worker ids. Legacy name: LIP_WORKERS."
    )
    asd: bool | None = Field(default=None, description="Enable the active-speaker gate. Legacy name: LIP_ASD.")
    asd_bs: int | None = Field(default=None, description="Active-speaker gate batch size. Legacy name: LIP_ASD_BS.")
    asd_conf: float | None = Field(
        default=None, description="SyncNet confidence needed to call a face speaking. Legacy name: LIP_ASD_CONF."
    )
    asd_offmax: int | None = Field(
        default=None,
        description="Maximum audio-video offset accepted by the gate, frames. Legacy name: LIP_ASD_OFFMAX.",
    )
    asd_static: float | None = Field(
        default=None, description="Motion threshold below which a face counts as static. Legacy name: LIP_ASD_STATIC."
    )
    asd_static_sync: bool | None = Field(
        default=None, description="Also run SyncNet on static faces. Legacy name: LIP_ASD_STATIC_SYNC."
    )
    empty_cache: bool | None = Field(
        default=None, description="Release the CUDA cache between chunks. Legacy name: LIP_EMPTY_CACHE."
    )
    gpu_frac: float | None = Field(
        default=None, description="Per-process CUDA memory fraction cap. Legacy name: LIP_GPU_FRAC."
    )
    multiface: bool | None = Field(
        default=None, description="Handle several faces per shot. Legacy name: LIP_MULTIFACE."
    )
    paste_v2: bool | None = Field(
        default=None, description="Use the ROI paste-back with colour matching. Legacy name: LIP_PASTE_V2."
    )
    latentsync_dir: str | None = Field(
        default=None,
        description="Path to the upstream LatentSync checkout (configs and checkpoints). Legacy name: LATENTSYNC_DIR.",
    )


class ControlSettings(ConfigGroup):
    """Control plane: actions, chat, rate limits and spend caps."""

    actions: bool | None = Field(
        default=None,
        description="Enable the validated control plane (actions, chat, events). Legacy name: DUB_ACTIONS.",
    )
    token: SecretStr | None = Field(
        default=None,
        description="Shared token for the control plane; generated into the state dir when unset. Legacy name: DUB_CTL_TOKEN.",
    )
    chat: bool | None = Field(default=None, description="Enable the chat planner. Legacy name: DUB_CHAT.")
    add_lang: bool | None = Field(
        default=None, description="Allow adding a target language mid-watch. Legacy name: DUB_ADD_LANG."
    )
    addlang: bool | None = Field(
        default=None, description="Control-plane switch for add-language actions. Legacy name: DUB_CTL_ADDLANG."
    )
    addlang_s: float | None = Field(
        default=None, description="Time budget for an add-language action, seconds. Legacy name: DUB_CTL_ADDLANG_S."
    )
    confirm_ttl: float | None = Field(
        default=None, description="Lifetime of a confirmation token, seconds. Legacy name: DUB_CTL_CONFIRM_TTL."
    )
    daily_synth_s: float | None = Field(
        default=None, description="Daily synthesis-seconds cap. Legacy name: DUB_CTL_DAILY_SYNTH_S."
    )
    daily_usd: float | None = Field(
        default=None, description="Daily chat spend cap, USD. Legacy name: DUB_CTL_DAILY_USD."
    )
    devhooks: bool | None = Field(
        default=None,
        description="Test-only: install development hooks (unsupported here). Legacy name: DUB_CTL_DEVHOOKS.",
    )
    dryrun: bool | None = Field(
        default=None, description="Run actions without side effects (tests). Legacy name: DUB_CTL_DRYRUN."
    )
    dryrun_delay: float | None = Field(
        default=None, description="Artificial delay for dry-run actions, seconds. Legacy name: DUB_CTL_DRYRUN_DELAY."
    )
    ev_keep: int | None = Field(default=None, description="Events kept per job. Legacy name: DUB_CTL_EV_KEEP.")
    ev_max_bytes: int | None = Field(
        default=None, description="Maximum bytes of the event log per job. Legacy name: DUB_CTL_EV_MAX_BYTES."
    )
    ev_replay: int | None = Field(
        default=None, description="Events replayed to a reconnecting client. Legacy name: DUB_CTL_EV_REPLAY."
    )
    gemini_per_min: int | None = Field(
        default=None, description="Global LLM calls per minute. Legacy name: DUB_CTL_GEMINI_PER_MIN."
    )
    global_synth: int | None = Field(
        default=None, description="Global concurrent syntheses. Legacy name: DUB_CTL_GLOBAL_SYNTH."
    )
    ip_actions_per_min: int | None = Field(
        default=None, description="Per-IP action requests per minute. Legacy name: DUB_CTL_IP_ACTIONS_PER_MIN."
    )
    ip_chat_per_min: int | None = Field(
        default=None, description="Per-IP chat requests per minute. Legacy name: DUB_CTL_IP_CHAT_PER_MIN."
    )
    ip_poll_per_min: int | None = Field(
        default=None, description="Per-IP poll requests per minute. Legacy name: DUB_CTL_IP_POLL_PER_MIN."
    )
    ip_read_per_min: int | None = Field(
        default=None, description="Per-IP read requests per minute. Legacy name: DUB_CTL_IP_READ_PER_MIN."
    )
    lang_max: int | None = Field(
        default=None, description="Maximum concurrently added languages. Legacy name: DUB_CTL_LANG_MAX."
    )
    lang_max_errs: int | None = Field(
        default=None,
        description="Chunk errors tolerated before an added language is abandoned. Legacy name: DUB_CTL_LANG_MAX_ERRS.",
    )
    lang_max_err_s: float | None = Field(
        default=None,
        description="Window over which chunk errors are counted, seconds. Legacy name: DUB_CTL_LANG_MAX_ERR_S.",
    )
    lang_poll: float | None = Field(
        default=None, description="Poll interval of the language scheduler, seconds. Legacy name: DUB_CTL_LANG_POLL."
    )
    lock_wait_s: float | None = Field(
        default=None, description="Maximum wait for a segment lock, seconds. Legacy name: DUB_CTL_LOCK_WAIT_S."
    )
    ls_poll: float | None = Field(
        default=None, description="Poll interval while waiting for lip sync, seconds. Legacy name: DUB_CTL_LS_POLL."
    )
    ls_wait: float | None = Field(
        default=None, description="Maximum wait for lip sync, seconds. Legacy name: DUB_CTL_LS_WAIT."
    )
    m2_timeout_s: float | None = Field(
        default=None, description="Timeout for an expression operation, seconds. Legacy name: DUB_CTL_M2_TIMEOUT_S."
    )
    max_synth: int | None = Field(
        default=None, description="Maximum concurrent control-plane syntheses. Legacy name: DUB_CTL_MAX_SYNTH."
    )
    previews_per_min: int | None = Field(
        default=None, description="Preview requests per minute. Legacy name: DUB_CTL_PREVIEWS_PER_MIN."
    )
    spend_cap_usd: float | None = Field(
        default=None, description="Per-job chat spend cap, USD. Legacy name: DUB_CTL_SPEND_CAP_USD."
    )
    sse_per_job: int | None = Field(
        default=None, description="Concurrent event streams per job. Legacy name: DUB_CTL_SSE_PER_JOB."
    )
    sse_total: int | None = Field(
        default=None, description="Concurrent event streams overall. Legacy name: DUB_CTL_SSE_TOTAL."
    )
    synth_cap_s: float | None = Field(
        default=None, description="Per-action synthesis-seconds cap. Legacy name: DUB_CTL_SYNTH_CAP_S."
    )


class ServerSettings(ConfigGroup):
    """HTTP server, admission and limits."""

    dub_root: str | None = Field(default=None, description="Directory holding all job state. Legacy name: DUB_ROOT.")
    port: int | None = Field(default=None, description="Port the server listens on. Legacy name: PORT.")
    max_concurrent: int | None = Field(
        default=None, description="Maximum concurrently running chunk processes. Legacy name: MAX_CONCURRENT."
    )
    token: SecretStr | None = Field(
        default=None, description="Optional shared secret required for mutating /api/ calls. Legacy name: DUB_TOKEN."
    )
    cors_origins: str | None = Field(
        default=None, description="Comma-separated allowed CORS origins. Legacy name: DUB_CORS_ORIGINS."
    )
    admission: bool | None = Field(
        default=None, description="Enable bounded job admission. Legacy name: DUB_ADMISSION."
    )
    fanout: int | None = Field(
        default=None, description="Chunks dispatched per language in parallel. Legacy name: DUB_FANOUT."
    )
    fault: str | None = Field(default=None, description="Test-only fault injection spec. Legacy name: DUB_FAULT.")
    max_dur_s: float | None = Field(
        default=None, description="Longest accepted source window, seconds. Legacy name: DUB_MAX_DUR_S."
    )
    max_edit_chars: int | None = Field(
        default=None, description="Longest accepted edited line, characters. Legacy name: DUB_MAX_EDIT_CHARS."
    )
    max_queued_jobs: int | None = Field(
        default=None, description="Maximum queued jobs. Legacy name: DUB_MAX_QUEUED_JOBS."
    )
    max_start_s: float | None = Field(
        default=None, description="Latest accepted window start, seconds. Legacy name: DUB_MAX_START_S."
    )
    max_targets: int | None = Field(
        default=None, description="Maximum target languages per job. Legacy name: DUB_MAX_TARGETS."
    )
    max_uncached_jobs: int | None = Field(
        default=None, description="Maximum concurrent jobs not yet cached. Legacy name: DUB_MAX_UNCACHED_JOBS."
    )
    max_video_s: float | None = Field(
        default=None, description="Longest accepted source video, seconds. Legacy name: DUB_MAX_VIDEO_S."
    )
    max_work_s: float | None = Field(
        default=None, description="Maximum work seconds admitted per job. Legacy name: DUB_MAX_WORK_S."
    )
    reconcile: bool | None = Field(
        default=None, description="Reconcile language-track state on startup. Legacy name: DUB_RECONCILE."
    )
    services: str | None = Field(
        default=None, description="Path to a services JSON used by the supervisor. Legacy name: DUB_SERVICES."
    )


class MirrorsSettings(ConfigGroup):
    """Optional local IndicTrans2 and separation services."""

    it2_max_body: int | None = Field(
        default=None, description="IndicTrans2 mirror: maximum request body, bytes. Legacy name: IT2_MAX_BODY."
    )
    it2_max_chars: int | None = Field(
        default=None, description="IndicTrans2 mirror: maximum characters per request. Legacy name: IT2_MAX_CHARS."
    )
    it2_max_pending: int | None = Field(
        default=None, description="IndicTrans2 mirror: maximum queued requests. Legacy name: IT2_MAX_PENDING."
    )
    it2_max_tokens: int | None = Field(
        default=None, description="IndicTrans2 mirror: maximum generated tokens. Legacy name: IT2_MAX_TOKENS."
    )
    it2_models_dir: str | None = Field(
        default=None,
        description="IndicTrans2 mirror: directory with the model checkpoints. Legacy name: IT2_MODELS_DIR.",
    )
    sep_root: str | None = Field(
        default=None, description="Separator: only paths under this directory are accepted. Legacy name: SEP_ROOT."
    )
    sep_gpu: str | None = Field(default=None, description="Separator: GPU selection. Legacy name: SEP_GPU.")
    sep_decode_parallel: int | None = Field(
        default=None, description="Separator: concurrent audio decodes. Legacy name: SEP_DECODE_PARALLEL."
    )
    sep_max_bytes: int | None = Field(
        default=None, description="Separator: maximum input size, bytes. Legacy name: SEP_MAX_BYTES."
    )
    sep_max_pending: int | None = Field(
        default=None, description="Separator: maximum queued requests. Legacy name: SEP_MAX_PENDING."
    )
    sep_queue_timeout: float | None = Field(
        default=None, description="Separator: maximum queue wait, seconds. Legacy name: SEP_QUEUE_TIMEOUT."
    )


class Settings(BaseSettings):
    """Root settings object; build with ``Settings()`` (reads the environment and ``.env``)."""

    model_config = SettingsConfigDict(
        env_prefix="SUNWAI_", env_nested_delimiter="__", env_file=".env", extra="ignore", frozen=True
    )

    bodhan: BodhanSettings = Field(default_factory=BodhanSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    planning: PlanningSettings = Field(default_factory=PlanningSettings)
    asr: AsrSettings = Field(default_factory=AsrSettings)
    mt: MtSettings = Field(default_factory=MtSettings)
    fit: FitSettings = Field(default_factory=FitSettings)
    tts: TtsSettings = Field(default_factory=TtsSettings)
    mixing: MixingSettings = Field(default_factory=MixingSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    lipsync: LipsyncSettings = Field(default_factory=LipsyncSettings)
    control: ControlSettings = Field(default_factory=ControlSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    mirrors: MirrorsSettings = Field(default_factory=MirrorsSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Order: init, canonical env, ``.env``, legacy flat variables, file secrets."""
        return (init_settings, env_settings, dotenv_settings, LegacySource(settings_cls), file_secret_settings)
