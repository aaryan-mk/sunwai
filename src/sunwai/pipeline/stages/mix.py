"""Mix stage: align, duck, mix, limit and mux the chunk."""

from __future__ import annotations

import logging
import math
import os
import threading as _threading
from pathlib import Path
from typing import Any, Dict, List

import librosa
import numpy as np

log = logging.getLogger(__name__)

from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.stage import StagePolicy
from sunwai.pipeline.models import to_records
from sunwai.pipeline.stages.base import Segments, StageBase

from sunwai.media.audio import save_wav_stereo
from sunwai.media.ffmpeg import run_ffmpeg
from sunwai.media.frames import _audio_track_dur, _chunk_owned_samples, _frames_to_samples
from sunwai.pipeline.mixing.alignment import MIX_ALIGN_PAUSES, _align_pauses, _orig_sentence_resumes
from sunwai.pipeline.mixing.levels import MIX_LEVEL_FIX, MIX_LIMITER, MIX_LIM_LOOKAHEAD_MS, MIX_LIM_RELEASE_MS, _peak_limit, _trim_tts
from sunwai.pipeline.mixing.preservation import MIX_DUCK, MIX_DUCK_ATTACK_MS, MIX_DUCK_DB, MIX_DUCK_HOLD_MS, MIX_DUCK_RELEASE_MS, MIX_KEEP_EVENTS, MIX_KEEP_OVERLAP, MIX_OVERLAP_MAX_CUT, _duck_gain, _event_spans, _guard_overlap, _overlap_spans, _span_mask, _speech_activity
from sunwai.pipeline.support.artifacts import mix_mask
from sunwai.pipeline.support.guards import media_ok
from sunwai.pipeline.support.storage import atomic_write_json


def stage_mix(args, out_dir: Path, segments: List[Dict[str, Any]]):
    out_wav = out_dir / "dub.wav"
    out_mp4 = out_dir / f"dub_{args.lang}.mp4"

    # Load original 44.1k stereo mix, bed, vocals
    orig, _ = librosa.load(str(out_dir / "src44k.wav"), sr=44100, mono=False, dtype="float32")
    if orig.ndim == 1:
        orig = np.stack([orig, orig])
    orig = orig.T  # T x 2
    N = orig.shape[0]

    if (out_dir / "bed.wav").exists():
        bed, _ = librosa.load(str(out_dir / "bed.wav"), sr=44100, mono=False, dtype="float32")
        bed = bed.T if bed.ndim == 2 else np.stack([bed, bed], axis=1)
    else:
        bed = orig.copy()
    if (out_dir / "vocals.wav").exists():
        vocals, _ = librosa.load(str(out_dir / "vocals.wav"), sr=44100, mono=False, dtype="float32")
        vocals = vocals.T if vocals.ndim == 2 else np.stack([vocals, vocals], axis=1)
    else:
        vocals = orig.copy()

    bed = bed[:N] if len(bed) >= N else np.pad(bed, ((0, N - len(bed)), (0, 0)))
    vocals = vocals[:N] if len(vocals) >= N else np.pad(vocals, ((0, N - len(vocals)), (0, 0)))

    sr = 44100
    cf = int(0.08 * sr)  # 80 ms crossfade samples

    def rms(x):
        return float(np.sqrt(np.mean(np.sum(x * x, axis=1)) + 1e-12))

    # Load every usable TTS wav ONCE. A segment is "dubbed" only if it has a TTS wav and is not a fallback; fallback /
    # failed segments keep the ORIGINAL audio (they are not in the mask), so their vocals are never silently removed.
    dubbed = []   # (s, e_mask, t_stereo_scaled)
    for seg in segments:
        if seg["error"] or seg.get("fallback"):
            continue
        s = max(0, min(int(seg["start"] * sr), N))
        e = max(0, min(int(seg["end"] * sr), N))
        if e - s <= 0:
            continue
        tts_path = out_dir / "tts" / f"{seg['id']}.wav"
        if not tts_path.exists():
            continue
        t, _ = librosa.load(str(tts_path), sr=44100, mono=True, dtype="float32")
        if len(t) == 0:
            continue
        # put the dub's speech onset on the original onset (trim TTS lead/trail silence, keep ~30 ms) and, for
        # multi-sentence slots, restart each sentence where the original speaker resumed. Guarded, fail-open.
        if MIX_ALIGN_PAUSES or os.environ.get("MIX_TRIM", "1") == "1":
            try:
                if os.environ.get("MIX_TRIM", "1") == "1" and not seg.get("sent_pieces"):      # pieces carry intentional silence between sentences
                    t, lead_kept = _trim_tts(t, sr)
                else:
                    lead_kept = 0.0
                if MIX_ALIGN_PAUSES and not seg.get("sent_pieces") and not seg.get("slack_added"):      # already source-placed / slack-padded in stage_tts
                    _nx = [x["start"] for x in segments if x["start"] > seg["start"] + 1e-6]
                    _room = (min(_nx) - seg["start"]) if _nx else None
                    rs = _orig_sentence_resumes(out_dir / "vad.json", seg["start"], seg["end"])
                    if rs:
                        t2, info = _align_pauses(t, sr, lead_kept, [r - seg["start"] for r in rs], seg["end"] - seg["start"], room=_room)
                        if info:
                            log.info("mix align seg %s: %s", seg["id"], info)
                            t = t2
                s = max(0, s - int(lead_kept * sr))
            except Exception as _e:
                log.warning("mix align seg %s skipped: %s", seg.get("id"), _e)
                t, _ = librosa.load(str(tts_path), sr=44100, mono=True, dtype="float32")
                s = max(0, min(int(seg["start"] * sr), N))
        ref_rms = rms(vocals[s:e])
        t_rms = float(np.sqrt(np.mean(t * t) + 1e-12))
        if MIX_LEVEL_FIX:
            ref_rms = ref_rms / math.sqrt(vocals.shape[1])     # per-channel power, same convention as the mono t_rms
        if MIX_LEVEL_FIX:        # validity on UNFLOORED power (the +1e-12 floor made rms() > 1e-8 always true)
            _vs = vocals[s:e]
            _rp = float(np.mean(np.sum(_vs * _vs, axis=1))) / vocals.shape[1] if len(_vs) else 0.0
            _valid = _rp > 1e-16 and float(np.mean(t * t)) > 1e-16
        else:
            _valid = ref_rms > 1e-8 and t_rms > 1e-8
        if _valid:
            scale = ref_rms / t_rms
            if MIX_LEVEL_FIX:                                   # real +-9 dB gain clamp (the old expression always contained `scale`)
                scale = max(10 ** (-9 / 20), min(10 ** (9 / 20), scale))
            else:
                scale = max(ref_rms / (t_rms * (10 ** (9/20))), min(ref_rms / (t_rms * (10 ** (-9/20))), scale))
            t = t * scale
        t_st = np.stack([t, t], axis=1)
        wl = min(len(t_st), N - s)
        # borrowed following silence: the TTS may run past the slot end -> extend the replaced region to the end of the
        # speech (+ one crossfade so the tail is not half-attenuated by the smoothing) instead of attenuating it away
        e_mask = max(e, s + wl)
        if s + wl > e - cf:
            e_mask = min(N, max(e, s + wl + cf))
        dubbed.append((s, e_mask, t_st[:wl]))

    mask_sm = mix_mask([(s, em) for (s, em, _) in dubbed], N, sr)
    # (all guarded, fail-open to the old mix; kill switches MIX_KEEP_OVERLAP / MIX_DUCK / MIX_KEEP_EVENTS)
    _g90 = {}
    dub_act = None
    if dubbed and (MIX_KEEP_OVERLAP or MIX_DUCK or MIX_KEEP_EVENTS):
        try:
            dub_act = _speech_activity(dubbed, N, sr)
        except Exception as _e:
            log.warning("mix g90c activity skipped: %s", _e)
    if MIX_KEEP_OVERLAP and dubbed:
        try:
            _ovs = _overlap_spans(out_dir, N, sr)
            if _ovs:
                _ovs = _guard_overlap(_ovs, dubbed, sr, MIX_OVERLAP_MAX_CUT)
            if _ovs:
                _ovm = mix_mask(_ovs, N, sr)
                mask_sm = (mask_sm * (1.0 - _ovm)).astype(np.float32)
                _g90["overlap_kept_s"] = round(sum(b - a for a, b in _ovs) / sr, 3)
        except Exception as _e:
            log.warning("mix overlap keep skipped: %s", _e)
    base_with_bed = bed
    if MIX_DUCK and dub_act is not None and dub_act.any():
        try:
            _gf = _duck_gain(dub_act, MIX_DUCK_DB, MIX_DUCK_ATTACK_MS, MIX_DUCK_RELEASE_MS, MIX_DUCK_HOLD_MS)
            _gs = _frames_to_samples(_gf, N, sr)
            base_with_bed = bed * _gs[:, None]
            _g90["duck_min_db"] = round(float(20 * np.log10(max(float(_gs.min()), 1e-6))), 2)
        except Exception as _e:
            log.warning("mix ducking skipped: %s", _e)
            base_with_bed = bed
    _kev = None
    if MIX_KEEP_EVENTS and dub_act is not None and (out_dir / "vocals.wav").exists():
        try:
            _evs, _evi = _event_spans(orig, vocals, mask_sm, dub_act, N, sr)
            _g90["events"] = _evi
            if _evs:
                _kev = _span_mask(_evs, N, sr)
                _g90["event_kept_s"] = round(sum(b - a for a, b in _evs) / sr, 3)
        except Exception as _e:
            log.warning("mix event keep skipped: %s", _e)
    final = orig.copy()
    for i in range(0, N, 4096):
        j = min(i + 4096, N)
        m = mask_sm[i:j]
        final[i:j] = final[i:j] * (1.0 - m[:, None]) + base_with_bed[i:j] * m[:, None]
        if _kev is not None:
            final[i:j] += vocals[i:j] * (_kev[i:j] * m)[:, None]
    if _g90:
        try:
            atomic_write_json(out_dir / "g90c.json", _g90)
        except Exception as _e:
            log.warning("g90c telemetry skipped: %s", _e)
    for (s, em, t_st) in dubbed:
        wl = len(t_st)
        m = mask_sm[s:s + wl]
        final[s:s + wl] += t_st * m[:, None]
    try:                                             # cheap per-chunk telemetry (placement overlap, pre-limiter peak), always on
        _sp = sorted((s, s + len(t_st)) for (s, em, t_st) in dubbed)
        _ov = [max(0, a_end - b_s) / sr for (a_s, a_end), (b_s, b_e) in zip(_sp, _sp[1:])]
        _fin = lambda v: float(v) if np.isfinite(v) else 0.0
        atomic_write_json(out_dir / "tts_placed.json", {
            "peak_pre_limit": _fin(np.max(np.abs(final))), "frac_over_0.89": _fin(np.mean(np.abs(final) > 0.89)),
            "overlap_s": _fin(sum(_ov)), "n_overlaps_gt_50ms": int(sum(1 for o in _ov if o > 0.05)),
            "max_overlap_s": _fin(max(_ov) if _ov else 0.0), "n_dubbed": len(dubbed)})
    except Exception as _e:
        log.warning("mix telemetry skipped: %s", _e)
    if os.environ.get("MIX_DUMP_STEM") == "1":      # measurement aid: placed TTS only; never fails the mix
        try:
            _stem = np.zeros_like(final)
            for (s, em, t_st) in dubbed:
                _stem[s:s + len(t_st)] += t_st
            _tmp = out_dir / f".stemtmp_{os.getpid()}.wav"
            save_wav_stereo(_tmp, _stem, 44100)
            os.replace(_tmp, out_dir / "tts_placed.wav")
        except Exception as _e:
            log.warning("stem dump skipped: %s", _e)

    # Peak limit -1 dBFS
    ceiling = 10 ** (-1 / 20)
    if MIX_LIMITER:
        if not np.isfinite(final).all():
            log.warning("mix: non-finite samples in the mix; zeroed")
            final = np.nan_to_num(final, nan=0.0, posinf=0.0, neginf=0.0)
        final = _peak_limit(final, ceiling, sr, MIX_LIM_LOOKAHEAD_MS, MIX_LIM_RELEASE_MS)
    else:
        peak = float(np.max(np.abs(final))) + 1e-12
        if peak > ceiling:
            final = final * (ceiling / peak)

    # Atomic publish: write temp files (with the right extension for ffmpeg), validate, then os.replace.
    # The mp4 is replaced LAST so its mtime is the newest (the lip-sync worker keys on it).
    tag = f"{os.getpid()}_{_threading.get_ident()}"
    tmp_wav = out_dir / f".dubtmp_{tag}.wav"
    tmp_mp4 = out_dir / f".dubtmp_{args.lang}_{tag}.mp4"
    try:
        # . Exact mode: dub.wav = the samples this chunk owns, mux WITHOUT `-shortest` (the old mux dropped the last 2-4 AAC
        # frames, 46-186 ms, of every chunk). The audio edit list then ends exactly at the owned length. Output is validated; any
        # mismatch/exception -> the legacy path (-shortest, full N samples). Outcome is logged and recorded in sync_metrics.json.
        # Video is NOT cut (stream copy cannot drop the extra last frame: it is a reference frame of displayed B-frames).
        import soundfile as _sf
        _why = None
        N_own, _exact, _why = _chunk_owned_samples(out_dir, N, sr)
        if _exact:
            try:
                _w = final[:N_own]
                save_wav_stereo(tmp_wav, _w, sr)
                _inf = _sf.info(str(tmp_wav))
                if _inf.frames != N_own or _inf.samplerate != sr:
                    raise RuntimeError(f"dub.wav validation failed: frames {_inf.frames} != {N_own}")
                run_ffmpeg(["-i", str(out_dir / "src.mp4"), "-i", str(tmp_wav), "-map", "0:v:0", "-map", "1:a:0",
                            "-c:v", "copy", "-c:a", "aac", str(tmp_mp4)])
                _ad = _audio_track_dur(tmp_mp4)
                if not media_ok(tmp_mp4, need_video=True, need_audio=True, min_dur=max(0.05, 0.9 * N_own / sr - 1.0)):
                    raise RuntimeError("dub mp4 validation failed (ffprobe)")
                if _ad is None or not math.isfinite(_ad) or abs(_ad - N_own / sr) > 0.003:
                    raise RuntimeError(f"audio track {_ad} s != owned {N_own / sr:.4f} s")
            except Exception as _e:
                _why = f"{type(_e).__name__}: {_e}"
                _exact = False
        if not _exact:
            N_own = N
            log.warning("mix: LEGACY mux path taken for this chunk (exact_own=false): %s", _why)
            save_wav_stereo(tmp_wav, final, sr)
            _inf = _sf.info(str(tmp_wav))
            if _inf.frames != N or _inf.samplerate != sr:
                raise RuntimeError(f"dub.wav validation failed: frames {_inf.frames} != {N}")
            run_ffmpeg(["-i", str(out_dir / "src.mp4"), "-i", str(tmp_wav), "-map", "0:v:0", "-map", "1:a:0",
                        "-c:v", "copy", "-c:a", "aac", "-shortest", str(tmp_mp4)])
        try:
            atomic_write_json(out_dir / "sync_metrics.json", {"exact_own": bool(_exact), "reason": _why, "n_samples": int(N),
                              "owned_samples": int(N_own), "audio_track_s": _audio_track_dur(tmp_mp4)})
        except Exception as _e:
            log.warning("sync_metrics.json not written: %s", _e)
        if not media_ok(tmp_mp4, need_video=True, need_audio=True, min_dur=max(0.05, 0.9 * N_own / sr - 1.0)):
            raise RuntimeError("dub mp4 validation failed (ffprobe)")
        os.replace(tmp_wav, out_wav)
        os.replace(tmp_mp4, out_mp4)
    finally:
        for _p in (tmp_wav, tmp_mp4):
            try:
                os.remove(_p)
            except OSError:
                pass
    log.info("Stage mix: done -> %s", out_mp4)
    return out_mp4


class MixStage(StageBase[Segments, Path]):
    """Align, duck, mix and limit the final audio, then mux the chunk video."""

    name = "mix"
    policy = StagePolicy(timing_key="mix")

    def run(self, value: Segments, ctx: ChunkContext) -> Path:
        """Align, duck, mix and limit the final audio, then mux the chunk video."""
        return stage_mix(ctx.legacy_args(), ctx.work_dir, to_records(value))  # type: ignore[no-any-return]
