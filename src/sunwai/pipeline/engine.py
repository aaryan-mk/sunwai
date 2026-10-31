#!/usr/bin/env python3
"""Command line entry point and compatibility facade of the dubbing pipeline.

``python -m sunwai.pipeline.engine --url ... --lang hi --out DIR`` dubs one chunk. The implementation lives in the
stage modules (``sunwai.pipeline.stages``), the shared helpers (``sunwai.media``, ``sunwai.language``,
``sunwai.synthesis``, ``sunwai.clients.service_calls``) and ``sunwai.pipeline.shared_prep``; this module re-exports
their public names so older imports such as ``engine.pick_voice`` keep working.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

from sunwai.language.spoken import _expr_spoken, _num_spoken  # noqa: F401
from sunwai.pipeline import runtime
from sunwai.clients.service_endpoints import (ASR_BASE, BASE_URL, IT2_BASE, IT2_TARGETS, MT_BASE, MT_BASES, TTS_BASE, TTS_BASES, TTS_MODEL, _TTS_RR, _next_mt_base, _next_tts_base)  # noqa: F401
from sunwai.clients.service_calls import (_bodhan_tts_bounded, _budget_timeout, _post_bounded, _stream_post, _with_retries, bodhan_asr, bodhan_mt, bodhan_tts, load_keys)  # noqa: F401
from sunwai.language.numbers import (  # noqa: F401
    _TN_CACHE,
    _TN_CACHE_MAX,
    _TN_CURRENCY_CODES,
    _TN_DOTTED_CUES,
    _TN_ID_CUES,
    _TN_INV,
    _TN_LOCK,
    _TN_NUM_LANGS,
    _TN_PCT,
    _TN_SKIP_ONE,
    _TN_UNIT_AFTER,
    _TN_VER,
    _tn_cue_before,
    _tn_digits,
    _tn_flag,
    _tn_int_words,
    _tn_inv,
    _tn_next_word,
    _tn_numbers,
    _tn_quantity_words,
    _tn_words_to_int,
    normalize_numbers,
)
from sunwai.media.audio import (  # noqa: F401
    _wav_bytes,
    _wav_ok,
    _wav_pcm,
    concat_wav_bytes,
    crop_to_seconds,
    load_wav_mono,
    load_wav_stereo,
    save_wav_mono,
    save_wav_stereo,
    trim_wav_bytes,
    wav_duration,
)
from sunwai.media.ffmpeg import (  # noqa: F401
    run_ffmpeg,
)
from sunwai.media.frames import (  # noqa: F401
    _audio_track_dur,
    _chunk_owned_samples,
    _frames_to_samples,
    _own_from_ts,
    _owned_frames_by_ts,
)
from sunwai.pipeline.mixing.alignment import (  # noqa: F401
    MIX_ALIGN_MIN_DRIFT,
    MIX_ALIGN_PAUSES,
    _align_pauses,
    _orig_sentence_resumes,
)
from sunwai.pipeline.mixing.levels import (  # noqa: F401
    MIX_KEEP_LEAD,
    MIX_KEEP_TAIL,
    MIX_LEVEL_FIX,
    MIX_LIMITER,
    MIX_LIM_LOOKAHEAD_MS,
    MIX_LIM_RELEASE_MS,
    MIX_TRIM_REL,
    _frame_db,
    _peak_limit,
    _pretrim_wav,
    _speech_bounds,
    _trim_tts,
)
from sunwai.pipeline.mixing.preservation import (  # noqa: F401
    MIX_DUCK,
    MIX_DUCK_ATTACK_MS,
    MIX_DUCK_DB,
    MIX_DUCK_HOLD_MS,
    MIX_DUCK_RELEASE_MS,
    MIX_EVENT_MAX_CAND,
    MIX_EVENT_MIN_S,
    MIX_EVENT_THR,
    MIX_KEEP_EVENTS,
    MIX_KEEP_OVERLAP,
    MIX_OVERLAP_MAX_CUT,
    _ACTIVITY_HOP,
    _AST,
    _EVENT_CLASSES,
    _ast_probs,
    _dub_activity_frames,
    _duck_gain,
    _event_spans,
    _guard_overlap,
    _overlap_spans,
    _preserve_flag,
    _span_mask,
    _speech_activity,
)
from sunwai.pipeline.resynth import (  # noqa: F401
    do_resynth,
)
from sunwai.pipeline.shared_prep import (  # noqa: F401
    GEN_FILE,
    _GEN_CORE,
    _NEWGEN_PENDING,
    _PREP_FILES,
    _SLOT_DEP,
    _chunk_mode,
    _early_chunk,
    _gen_inputs,
    _gen_inputs_changed,
    _gen_marker,
    _gen_read,
    _gen_verify,
    _prep_file_ok,
    _prep_shared_new,
    _prep_stages,
    _publish_files,
    _purge_slot_dep,
    chunk_lookahead_gate,
    prep_shared,
)
from sunwai.pipeline.stages.asr import (  # noqa: F401
    ASR_DEDUP,
    ASR_LONG_SPLIT,
    _WORD_RE,
    _asr_words,
    _reset_downstream,
    dedup_boundary_words,
    stage_asr,
)
from sunwai.pipeline.stages.audio import (  # noqa: F401
    stage_audio,
)
from sunwai.pipeline.stages.fetch import (  # noqa: F401
    stage_fetch,
)
from sunwai.pipeline.stages.mix import (  # noqa: F401
    stage_mix,
)
from sunwai.pipeline.stages.mt import (  # noqa: F401
    MT_PIECE_WORDS,
    MT_RECOVER_BUDGET_S,
    MT_RECOVER_CALL_S,
    MT_RECOVER_CAP_S,
    MT_RECOVER_CHUNK_S,
    MT_RECOVER_EXTEND_S,
    MT_RECOVER_MAX_PIECES,
    _MT_SPENT,
    _MT_SPENT_LOCK,
    _MtBudget,
    _is_timeout,
    _it2_mt,
    _mt_pieces,
    _mt_pool_grant,
    _mt_script_recover,
    _mt_script_recover_inner,
    routed_mt,
    stage_mt,
)
from sunwai.pipeline.stages.numbers import (  # noqa: F401
    stage_numbers,
)
from sunwai.pipeline.stages.separate import (  # noqa: F401
    stage_separate,
)
from sunwai.pipeline.stages.slots import (  # noqa: F401
    CROP_HALFGAP,
    FORCE_CUT_MAX_S,
    SPLIT_MIN_PIECE_S,
    SPLIT_MIN_SPK_S,
    SPLIT_SPK,
    _NEWGEN,
    _cut_points,
    _force_cut,
    _zero_slots,
    split_slots_by_speaker,
    stage_slots,
)
from sunwai.pipeline.stages.speakers import (  # noqa: F401
    ASR_LANGS,
    LID_LOCK,
    _lid_compute,
    _lid_first_writer,
    _lid_is_leader,
    _read_lid_record,
    _source_lang,
    detect_source_language,
    stage_speakers,
)
from sunwai.pipeline.stages.tts import (  # noqa: F401
    stage_tts,
)
from sunwai.pipeline.stages.vad import (  # noqa: F401
    stage_vad,
)
from sunwai.pipeline.support.artifacts import (  # noqa: F401
    mix_mask,
    raw_tts_fresh,
    tts_fingerprint,
    write_raw_tts,
)
from sunwai.pipeline.support.concurrency import (  # noqa: F401
    _LIMITERS,
    _SlotLimiter,
    _TLS,
    _fault,
    inflight,
    thread_session,
)
from sunwai.pipeline.support.env import (  # noqa: F401
    _env_clamped,
    _env_num,
    _env_workers,
    _envf,
    _flag,
)
from sunwai.pipeline.support.expression_flags import (  # noqa: F401
    _EXPR_FLAG,
    _expr_flag,
    _expr_on,
    _expr_style,
)
from sunwai.pipeline.support.guards import (  # noqa: F401
    enforce_fallback_budget,
    failed_speech_frac,
    mark_fallback,
    media_ok,
)
from sunwai.pipeline.support.storage import (  # noqa: F401
    _atomic_copy,
    _safe_cleanup,
    _sha_file,
    atomic_write_json,
    load_timings,
    save_timing,
)
from sunwai.pipeline.voices import (  # noqa: F401
    DEFAULT_VOICE,
    LANGS,
    VOICE_MAP,
    _PLANNED_VOICES,
    _POOLS,
    _pick_voice_legacy,
    _voice_pools,
    pick_voice,
    voice_preflight,
    voice_require_speakers,
)
from sunwai.synthesis.pacing import (  # noqa: F401
    _TTS_LAT,
    _TTS_LAT_LOCK,
    _record_lat,
    _tts_slow_min,
    _tts_slow_now,
)
from sunwai.synthesis.runaway import (  # noqa: F401
    _runaway_gps,
    is_runaway,
    runaway_ladder,
    runaway_ladder_on,
    runaway_split_pieces,
)
from sunwai.synthesis.splitting import (  # noqa: F401
    _HOP,
    _MISSING,
    _SPLIT_SEM,
    _SPLIT_SEM_LOCK,
    _SPLIT_STRONG,
    _SPLIT_WEAK,
    _SplitRestoreFailed,
    _g_count,
    _split_sem,
    _split_target,
    _src_subspans,
    _stretch_pauses,
    _whole_pause_drift,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(args, out_dir: Path, segments: List[Dict[str, Any]], out_mp4: Path):
    n = len(segments)
    spk_g: Dict[int, str] = {}
    for s in segments:
        spk_g[s["speaker"]] = s["gender"]
    ratios = [s.get("ratio", 1.0) for s in segments if not s["error"]]
    mean_r = float(np.mean(ratios)) if ratios else 0.0
    max_r = float(np.max(ratios)) if ratios else 0.0
    ovf = sum(1 for s in segments if s.get("overflow"))
    timings = load_timings(out_dir)
    log.info("=== SUMMARY ===")
    log.info("segments: %d", n)
    for spk, g in sorted(spk_g.items()):
        log.info("  speaker %d: %s", spk, g)
    log.info("fit ratio mean=%.3f max=%.3f", mean_r, max_r)
    log.info("overflow segments: %d", ovf)
    for stage in ["fetch", "audio", "vad", "slots", "speakers", "asr", "mt", "tts", "separate", "mix"]:
        if stage in timings:
            log.info("  %-9s: %.2fs", stage, timings[stage])
    log.info("output: %s", out_mp4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--lang", required=True, choices=sorted(LANGS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--dur", type=float, default=90.0)
    parser.add_argument("--resynth", type=int, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    voice_preflight(args.lang, out_dir)           # outside every segment-level handler: a bad plan fails the chunk

    # stash duration for slot force-cut
    runtime.args_dur = args.dur

    if args.resynth is not None:
        if not args.text:
            parser.error("--resynth requires --text")
        do_resynth(args, out_dir)
        print("ALL DONE")
        return

    # Full pipeline
    chunk_lookahead_gate(out_dir)
    segments = prep_shared(args, out_dir)
    t0 = time.time(); segments = stage_mt(args, out_dir, segments); save_timing(out_dir, "mt", time.time() - t0)
    from sunwai.pipeline.fitting.fit_stage import stage_fit
    args.pick_voice = pick_voice                       # stage_fit budgets per voice
    t0 = time.time(); segments = stage_fit(args, out_dir, segments); save_timing(out_dir, "fit", time.time() - t0)
    t0 = time.time(); segments = stage_numbers(args, out_dir, segments); save_timing(out_dir, "numbers", time.time() - t0)
    t0 = time.time(); segments = stage_tts(args, out_dir, segments); save_timing(out_dir, "tts", time.time() - t0)
    t0 = time.time(); stage_separate(out_dir); save_timing(out_dir, "separate", time.time() - t0)
    t0 = time.time(); out_mp4 = stage_mix(args, out_dir, segments); save_timing(out_dir, "mix", time.time() - t0)

    print_summary(args, out_dir, segments, out_mp4)
    print("ALL DONE")
