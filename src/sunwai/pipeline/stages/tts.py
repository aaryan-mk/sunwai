"""TTS stage: synthesis with sentence splitting, pause stretching and runaway repair."""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading as _threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import librosa
import numpy as np

from sunwai.paths import DATA_DIR

log = logging.getLogger(__name__)

from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.stage import StagePolicy
from sunwai.pipeline.models import from_records, to_records
from sunwai.pipeline.stages.base import Segments, StageBase

from sunwai.clients.service_endpoints import TTS_BASES
from sunwai.clients.service_calls import _bodhan_tts_bounded, bodhan_tts, load_keys
from sunwai.media.audio import trim_wav_bytes, wav_duration
from sunwai.media.ffmpeg import run_ffmpeg
from sunwai.pipeline.mixing.levels import _pretrim_wav
from sunwai.pipeline.shared_prep import _early_chunk
from sunwai.pipeline.support.artifacts import raw_tts_fresh, tts_fingerprint, write_raw_tts
from sunwai.pipeline.support.concurrency import _fault, thread_session
from sunwai.pipeline.support.env import _env_workers, _envf, _flag
from sunwai.language.spoken import _expr_spoken
from sunwai.pipeline.support.expression_flags import _expr_on, _expr_style
from sunwai.pipeline.support.guards import enforce_fallback_budget, mark_fallback
from sunwai.pipeline.support.storage import _safe_cleanup, atomic_write_json
from sunwai.pipeline.voices import pick_voice, voice_require_speakers
from sunwai.synthesis.pacing import _record_lat, _tts_slow_min, _tts_slow_now
from sunwai.synthesis.runaway import is_runaway, runaway_ladder, runaway_ladder_on
from sunwai.synthesis.splitting import _MISSING, _SplitRestoreFailed, _split_sem, _split_target, _src_subspans, _stretch_pauses, _whole_pause_drift


def stage_tts(args, out_dir: Path, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    voice_require_speakers(segments)
    tts_dir = out_dir / "tts"
    tts_dir.mkdir(exist_ok=True)
    keys = load_keys()
    session = None  # per-thread sessions via thread_session()
    tts_key = keys["tts"]

    def synth(seg):
        """Any exception while synthesising/fitting one segment degrades THAT segment to the original audio."""
        try:
            return _synth(seg)
        except Exception as e:
            log.error("TTS stage error seg %s: %s", seg.get("id"), e)
            mark_fallback(seg, "tts", e)
            seg.update(ratio=1.0, tempo=1.0, overflow=False, overflow_s=0.0)
            for _n in (f"{seg['id']}.wav", f"{seg['id']}_raw.wav"):
                try:
                    os.remove(tts_dir / _n)
                except OSError:
                    pass
            return seg["id"]

    def _tts_best(text, voice, span, t_end=None):
        """one TTS call (+ at most one runaway retry, same policy as the whole-slot path); returns wav bytes."""
        import io as _io, wave as _wave
        best, _t0 = None, time.time()
        for attempt in range(2):                                   # per piece: first call + at most ONE runaway retry
            if t_end is not None and time.time() > t_end:
                if best is None:
                    raise TimeoutError("split-slot deadline")
                break
            _t0c = time.time()
            wb = _bodhan_tts_bounded(thread_session(), tts_key, text, args.lang, voice, t_end)
            _record_lat(time.time() - _t0c, wb)
            w = _wave.open(_io.BytesIO(wb))
            d = w.getnframes() / w.getframerate()
            if best is None or d < best[1]:
                best = (wb, d)
            if d / max(span, 1e-6) <= 2.5 or time.time() - _t0 > float(os.environ.get("PIPE_RUNAWAY_MAX_S", "30")) or (t_end is not None and time.time() > t_end):
                break
        if best[1] < 0.05:
            raise RuntimeError("TTS returned empty audio")
        if runaway_ladder_on() and is_runaway(best[1], span, text):        # a runaway piece is trimmed, never placed
            log.error("TTS runaway [piece] %.1fs for %.1fs span -> trim", best[1], span)
            return trim_wav_bytes(best[0], span * 1.25 + 0.05)
        return best[0]

    def _synth_pieces(seg, voice, out_path, state) -> bool:
        """slot with an internal source pause >= TTS_SENT_MIN_GAP: synthesise each source-aligned piece on its own and
        place it at its source sub-span, writing ONE slot wav (the mix is unchanged). False -> caller uses the whole-slot path."""
        import soundfile as _sf
        cid = seg["id"]
        if _expr_on(seg):
            state["skip"] = "expr"
            return False                      # an expression line is synthesised whole-slot (tts_text + style), never split
        spans = _src_subspans(out_dir, seg)
        if not spans:
            state["skip"] = "no_pause"
            return False
        plan = _split_target(seg["tgt_text"], spans)
        if not plan:
            state["skip"] = "no_split_point"
            return False
        _dr = _whole_pause_drift(out_path, spans, float(seg["start"]))      # the legacy audio is already on disk: only upgrade when its pauses are really off
        if _dr is None:
            state["skip"] = "drift_unknown"                   # cannot tell whether the legacy pauses are off: keep legacy
            return False
        if _dr < _envf("TTS_SENT_MIN_DRIFT", 0.25, 0.0, 5.0):
            state["skip"] = "aligned"
            return False
        s0, slot_dur = float(seg["start"]), max(seg["end"] - seg["start"], 1e-6)
        room = slot_dur
        for s2 in segments:
            if s2["id"] == cid + 1:
                room = max(slot_dur, s2["start"] - seg["start"])
        sr = 24000
        slow_min, cursor, placed, info = _tts_slow_min(), 0.0, [], []
        t_end = time.time() + _envf("TTS_SENT_DEADLINE_S", 25.0, 5.0, 120.0)
        made = state.setdefault("made", set())          # files created during THIS attempt (pre-cached raws are never deleted)
        def _giveup(why):
            state["skip"] = why.split(" ")[0].replace(":", "")
            _safe_cleanup(cid, tts_dir, made)
            log.info("per-sentence TTS seg %s: %s; whole-slot path", cid, why)
            return False
        for k, (txt, a, b) in enumerate(plan):
            if time.time() > t_end:
                return _giveup("deadline")
            if _tts_slow_now():
                return _giveup("slow")
            raw = tts_dir / f"{cid}_p{k}_raw.wav"
            fp = tts_fingerprint(txt, voice, args.lang)
            span = max(b - a, 1e-3)
            if not raw_tts_fresh(raw, fp):
                _fault("tts", cid)
                try:
                    wbp = _tts_best(txt, voice, span, t_end)
                except TimeoutError:
                    return _giveup("deadline")
                if time.time() > t_end:                       # the call crossed the deadline: drop the split, one whole-slot attempt
                    return _giveup("deadline")
                write_raw_tts(raw, wbp, fp)
                made.add(raw)                                 # tracked only once this attempt has really written it
            trim = tts_dir / f"{cid}_p{k}_trim.wav"
            fit_src, d = _pretrim_wav(raw, trim)
            if k < len(plan) - 1:
                allowed = (plan[k + 1][1] - a) - 0.10
            else:
                allowed = room - (a - s0)
            ratio = d / span
            tempo = min(ratio, 1.25) if ratio > 1.0 else (max(ratio, slow_min) if ratio < 1.0 else 1.0)
            if allowed <= 0 or d / 1.25 > allowed + 0.02:     # cannot meet its sub-span even at 1.25x: never delay later pieces
                return _giveup("misfit")
            if d / tempo > allowed > 0:                       # still too long: speed up further into the pause that follows
                tempo = min(max(tempo, d / allowed), 1.25)
            tmpw = tts_dir / f"{cid}_p{k}_fit.wav"
            run_ffmpeg(["-i", str(fit_src), "-filter:a", f"atempo={tempo:.6f}", "-ac", "1", "-ar", str(sr), str(tmpw)])
            y, _sr = _sf.read(str(tmpw), dtype="float32")
            for _p in (tmpw, trim):
                try:
                    os.remove(_p)
                except OSError:
                    pass
            start = max(a - s0, (cursor + 0.10) if k > 0 else 0.0)
            cursor = start + len(y) / sr
            placed.append((start, y))
            info.append(dict(t0=round(a - s0, 3), t1=round(b - s0, 3), start=round(start, 3), dur=round(len(y) / sr, 3), tempo=round(tempo, 3), text=txt))
        if cursor > room + 0.05:                              # does not fit even with the pauses squeezed: let the whole-slot path (refit) decide
            return _giveup("room")
        out = np.zeros(int(cursor * sr) + 1, dtype=np.float32)
        fl = int(0.004 * sr)
        for start, y in placed:
            y = y.copy()
            if len(y) > 2 * fl:
                y[:fl] *= np.linspace(0, 1, fl, dtype=y.dtype); y[-fl:] *= np.linspace(1, 0, fl, dtype=y.dtype)
            i0 = int(start * sr)
            out[i0:i0 + len(y)] += y[:len(out) - i0]
        tmp = out_path.with_name(f".{out_path.stem}.tmp{os.getpid()}_{_threading.get_ident()}.wav")
        made.add(tmp)                                  # removed by the guarded cleanup if the write/replace fails
        _sf.write(str(tmp), out, sr, subtype="PCM_16")
        state["replaced"] = True                       # pre-armed BEFORE the replace: from here on only the backup can bring the legacy audio back
        os.replace(tmp, out_path)
        seg["sent_pieces"] = info
        seg.pop("slack_added", None)
        seg["ratio"] = cursor / slot_dur
        seg["tempo"] = max(p["tempo"] for p in info)
        seg["overflow"] = False
        seg["overflow_s"] = 0.0
        seg["voice"] = voice
        log.info("per-sentence TTS seg %s: %d pieces, end %.2fs / slot %.2fs", cid, len(info), cursor, slot_dur)
        return True

    def _runaway_guard(seg, voice, raw_path, where):
        """uniform runaway check on the raw TTS file that is about to be fitted (fresh, CACHED, or after a refit). A runaway is never
        placed: split -> trim (shared 60 s deadline); if neither works the exception degrades this segment to the original audio."""
        if not runaway_ladder_on():
            return False
        cid = seg["id"]
        slot = max(seg["end"] - seg["start"], 1e-6)
        wb = Path(raw_path).read_bytes()
        d = wav_duration(wb)
        text = _expr_spoken(seg)
        if not is_runaway(d, slot, text):
            return False
        log.warning("TTS runaway [%s] seg %d: %.1fs for %.1fs slot", where, cid, d, slot)
        _end = time.time() + float(os.environ.get("PIPE_RUNAWAY_LADDER_S", "60"))          # ONE deadline for the whole ladder
        if _expr_on(seg):
            _synth_piece = lambda _p, _span: (_ for _ in ()).throw(RuntimeError("expression line: trim only"))
        else:
            _synth_piece = lambda _p, _span: _tts_best(_p, voice, _span, _end)
        nb, nd, how = runaway_ladder(text, slot, wb, d, _synth_piece, log)
        if how == "none":
            for _p in (Path(raw_path), Path(raw_path).with_suffix(".fp")):      # never leave the runaway cached: the next run re-synthesises
                try:
                    os.remove(_p)
                except OSError:
                    pass
            raise RuntimeError(f"runaway TTS ({d:.1f}s for a {slot:.1f}s slot) could not be repaired")
        log.error("TTS runaway [%s] seg %d: %.1fs for %.1fs slot -> ladder '%s' (%.1fs)", where, cid, d, slot, how, nd)
        write_raw_tts(Path(raw_path), nb, tts_fingerprint(text, voice, args.lang, _expr_style(seg)))
        seg["runaway_fix"] = how
        return True

    def _synth(seg):
        cid = seg["id"]
        state = {"one_try": False}
        out_path = tts_dir / f"{cid}.wav"
        raw_path = tts_dir / f"{cid}_raw.wav"
        if not (seg.get("tgt_text") or "").strip():
            seg.pop("sent_pieces", None); seg.pop("slack_added", None); seg.pop("legacy_fit", None); seg.pop("sent_skip", None)
            seg.update(ratio=1.0, tempo=1.0, overflow=False, overflow_s=0.0)
            for _p in (out_path, raw_path):
                try:
                    os.remove(_p)      # never leave a stale wav that the mix would pick up
                except OSError:
                    pass
            return cid   # nothing to speak (empty source): no TTS, original audio is kept
        _voice0 = pick_voice(args.lang, seg["gender"], seg["speaker"])
        seg.pop("sent_pieces", None)
        seg.pop("sent_skip", None)
        seg.pop("legacy_fit", None)
        _fp0 = tts_fingerprint(_expr_spoken(seg), _voice0, args.lang, _expr_style(seg))
        if not raw_tts_fresh(raw_path, _fp0):
            try:
                voice = _voice0
                _fault("tts", cid)
                # runaway guard: indic-speak occasionally keeps generating. The TTS server already re-samples runaways
                # (TTS_RUNAWAY_RETRIES), so the pipeline adds at most ONE more try (PIPE_RUNAWAY_TRIES=2) and only when the
                # first call was quick -> no stacked retries / bounded tail latency. Keep the shortest.
                import io as _io, wave as _wave
                slot = max(seg["end"] - seg["start"], 1e-6)
                best = None
                _t0 = time.time()
                for attempt in range(max(1, int(os.environ.get("PIPE_RUNAWAY_TRIES", "2")))):      # EXACT legacy whole-slot path (also after a split failure)
                    _t0c = time.time()
                    wb = bodhan_tts(thread_session(), tts_key, _expr_spoken(seg), args.lang, voice, style=_expr_style(seg))      # the ORIGINAL legacy call
                    _record_lat(time.time() - _t0c, wb)
                    w = _wave.open(_io.BytesIO(wb))
                    d = w.getnframes() / w.getframerate()
                    if best is None or d < best[1]:
                        best = (wb, d)
                    if d / slot <= 2.5 or time.time() - _t0 > float(os.environ.get("PIPE_RUNAWAY_MAX_S", "30")):
                        break
                    log.warning("TTS runaway seg %d: %.1fs for %.1fs slot (attempt %d)", cid, d, slot, attempt + 1)
                if best[1] < 0.05:
                    raise RuntimeError("TTS returned empty audio")
                write_raw_tts(raw_path, best[0], _fp0)
            except Exception as e:
                log.error("TTS error seg %d: %s", cid, e)
                mark_fallback(seg, "tts", e)
                seg["ratio"] = 1.0
                seg["tempo"] = 1.0
                seg["overflow_s"] = 0.0
                seg["voice"] = _voice0
                for _p in (out_path, raw_path):
                    try:
                        os.remove(_p)      # a stale wav must never be mixed in place of the failed one
                    except OSError:
                        pass
                return cid

        # compute durations
        seg.pop("runaway_fix", None)
        _runaway_guard(seg, _voice0, raw_path, "whole-slot/cache")
        trim_path = tts_dir / f"{cid}_trim.wav"
        fit_src, tts_dur = _pretrim_wav(raw_path, trim_path)
        slot_dur = max(seg["end"] - seg["start"], 1e-6)
        # refit: if even max tempo (1.25x) cannot fit the room up to the next segment, rewrite shorter at the measured rate
        room = slot_dur
        for s2 in segments:
            if s2["id"] == cid + 1:
                room = max(slot_dur, s2["start"] - seg["start"])
        if tts_dur > room * 1.25 + 0.05 and not seg.get("refit") and not seg.get("edited") and not _expr_on(seg) and os.environ.get("REFIT", "1") == "1" and not _early_chunk(out_dir):
            try:
                from sunwai.pipeline.fitting.fit_stage import graphemes, guarded_rewrite, source_lang_name, confident_source_lang
                g = graphemes(seg["tgt_text"])
                budget = max(4, int(g / tts_dur * room * 1.2))
                name = next((t.get("name") for t in json.load(open(DATA_DIR / "languages.json", encoding="utf-8"))["targets"] if t["code"] == args.lang), args.lang)
                new = guarded_rewrite(seg["src_text"], seg["tgt_text"], budget, name, args.lang, __import__("sunwai.clients.llm", fromlist=["x"]).legacy_gemini_key(),
                                      source_lang_name(out_dir), tag=f"refit seg {cid}", src_lang=confident_source_lang(out_dir))
                if new:        # script + word-split + shorter + meaning guard + one retry (same as stage_fit)
                    voice = pick_voice(args.lang, seg["gender"], seg["speaker"])
                    wb = bodhan_tts(thread_session(), tts_key, new, args.lang, voice, style=_expr_style(seg))
                    tmp = tts_dir / f"{cid}_refit.wav"
                    tmp.write_bytes(wb)
                    d2 = librosa.get_duration(path=str(tmp))
                    if os.environ.get("FIT_TRIM", "1") == "1" and os.environ.get("MIX_TRIM", "1") == "1":
                        _p2, d2 = _pretrim_wav(tmp, tts_dir / f"{cid}_trim2.wav")      # same basis as tts_dur
                        try:
                            os.remove(tts_dir / f"{cid}_trim2.wav")
                        except OSError:
                            pass
                    if d2 < tts_dur:
                        log.info("refit seg %d: %.1fs -> %.1fs (room %.1fs, budget %d g)", cid, tts_dur, d2, room, budget)
                        seg["pre_refit_text"], seg["tgt_text"], seg["refit"] = seg["tgt_text"], new, True
                        for _k in ("num_text", "num_src", "num_ver", "num_lang"):
                            seg.pop(_k, None)          # v3: the refit text is spoken as written
                        write_raw_tts(raw_path, wb, tts_fingerprint(new, voice, args.lang, _expr_style(seg)))
                        os.remove(tmp) if tmp.exists() else None
                        fit_src, tts_dur = _pretrim_wav(raw_path, trim_path)
            except Exception as e:
                log.warning("refit seg %d failed: %s", cid, e)
        if _runaway_guard(seg, pick_voice(args.lang, seg["gender"], seg["speaker"]), raw_path, "refit"):
            fit_src, tts_dur = _pretrim_wav(raw_path, trim_path)
        ratio = tts_dur / slot_dur
        if ratio > 1.0:
            tempo = min(ratio, 1.25)
        elif ratio < 0.8:
            tempo = 1.0
        else:
            tempo = 1.0
        if ratio < 1.0 and not seg.get("edited"):      # dubs ended early; slow down gently (floor TTS_SLOW_MIN, default 0.9) to fill the slot
            tempo = max(ratio, _tts_slow_min())
        # apply tempo
        run_ffmpeg([
            "-i", str(fit_src),
            "-filter:a", f"atempo={tempo:.6f}",
            "-ac", "1", "-ar", "24000",
            str(out_path),
        ])
        if fit_src is not raw_path:
            try:
                os.remove(fit_src)
            except OSError:
                pass
        fitted_dur = librosa.get_duration(path=str(out_path))
        seg.pop("slack_added", None)
        if (_flag("TTS_SLACK_PAUSES") and not seg.get("edited") and ratio < 1.0 and fitted_dur < slot_dur - 0.15):
            try:                                        # the rest of the slack goes into the TTS's own phrase pauses
                import soundfile as _sf
                _y, _sr0 = _sf.read(str(out_path), dtype="float32")
                _y2, _add = _stretch_pauses(_y, _sr0, slot_dur - fitted_dur)
                if _add > 0.02 and not np.isfinite(_y2).all():
                    raise RuntimeError("slack rewrite produced non-finite samples")
                if _add > 0.02:
                    _tmp = out_path.with_name(f".{out_path.stem}.slack{os.getpid()}_{_threading.get_ident()}.wav")
                    try:
                        _sf.write(str(_tmp), _y2, _sr0, subtype="PCM_16")
                        _chk, _sr1 = _sf.read(str(_tmp), dtype="float32")            # validate: readable, finite, duration within +-5%
                        _exp = fitted_dur + _add
                        if not (np.isfinite(_chk).all() and _sr1 == _sr0 and abs(len(_chk) / _sr1 - _exp) <= 0.05 * _exp):
                            raise RuntimeError("slack rewrite failed validation")
                        os.replace(_tmp, out_path)
                        fitted_dur = len(_chk) / _sr1
                        seg["slack_added"] = round(_add, 3)
                    finally:
                        try:
                            os.remove(_tmp)
                        except OSError:
                            pass
            except Exception as _e:
                log.warning("slack pauses seg %s skipped: %s", cid, _e)
        # overflow: still longer than slot_dur
        allowed = slot_dur
        # allowed spill into following gap up to next segment start
        nxt = None
        for s in segments:
            if s["id"] == cid + 1:
                nxt = s
                break
        if nxt is not None:
            allowed = max(slot_dur, nxt["start"] - seg["start"])
        if fitted_dur > allowed + 1e-3:
            overflow_s = fitted_dur - allowed
            seg["overflow"] = True
            # truncate with 50 ms fade to allowed length
            run_ffmpeg([
                "-i", str(out_path),
                "-t", f"{allowed}",
                "-af", f"afade=t=out:st={max(0,allowed-0.05)}:d=0.05",
                "-ac", "1", "-ar", "24000",
                str(out_path),
            ])
        else:
            seg["overflow"] = False
            overflow_s = 0.0

        seg["ratio"] = ratio
        seg["tempo"] = tempo
        seg["overflow_s"] = overflow_s
        seg["voice"] = pick_voice(args.lang, seg["gender"], seg["speaker"])
        try:
            _split_upgrade(seg, _voice0, out_path, state)
        except _SplitRestoreFailed:
            raise                                               # incoherent slot: let synth() mark the fallback
        except Exception:
            pass
        return cid

    _FIT_KEYS = ("ratio", "tempo", "overflow", "overflow_s", "voice", "slack_added", "sent_pieces", "legacy_fit")

    def _split_upgrade(seg, voice, out_path, state):
        """the exact legacy whole-slot audio is ALREADY on disk. Optionally replace it by per-sentence audio when its pauses are
        off. Skip / failure / deadline / exception => legacy audio AND legacy segment fields are exactly as before. The legacy file is
        kept as a backup until the replacement and its metadata are committed; on ANY error after the replace (including
        KeyboardInterrupt/SystemExit, which are re-raised after the rollback) the audio is restored first and the fit fields only
        after a verified audio restore. If the audio cannot be restored the segment is marked as a TTS fallback (original audio kept
        by the mix) and the backup is kept: never legacy metadata over split audio, never a deleted only-copy."""
        state_ = {"got": False, "bak": None, "safe": True}   # safe: the backup may be deleted
        snap = {k: (seg[k] if k in seg else _MISSING) for k in _FIT_KEYS}

        def rollback(e):
            state_["safe"] = False
            bak = state_["bak"]
            restored = (not state.get("replaced")) or (bak is not None and bak.exists())      # a missing backup only counts as safe BEFORE the replace
            need_copy = bool(state.get("replaced"))
            if need_copy and restored:
                restored = False
                for how in (os.replace, shutil.copyfile):         # second chance: copy instead of rename
                    try:
                        how(bak, out_path); restored = True; break
                    except Exception as e2:
                        try:
                            log.error("per-sentence TTS seg %s: legacy audio restore failed (%s)", seg.get("id"), e2)
                        except Exception:
                            pass
            elif need_copy:
                restored = False                                  # replaced but no backup to restore from
            if restored:
                state_["safe"] = True
                for k, v in snap.items():
                    if v is _MISSING:
                        seg.pop(k, None)
                    else:
                        seg[k] = v
                seg["sent_skip"] = "error"
            else:                                               # audio and legacy metadata can no longer agree: coherent failed-slot state, then propagate
                for k in ("sent_pieces", "slack_added", "legacy_fit", "sent_skip"):
                    seg.pop(k, None)
                seg.update(ratio=1.0, tempo=1.0, overflow=False, overflow_s=0.0)
                try:
                    os.remove(out_path)
                except OSError:
                    pass
                try:
                    log.error("per-sentence TTS seg %s: legacy restore failed; backup kept at %s", seg.get("id"), bak)
                except Exception:
                    pass
                raise _SplitRestoreFailed(f"legacy audio restore failed: {e}")      # synth() marks the slot as a TTS fallback (original audio kept)

        try:
            if not _flag("TTS_SENT_SPLIT", "1"):
                return
            if seg.get("edited"):
                seg["sent_skip"] = "edited"; return
            if _early_chunk(out_dir):
                seg["sent_skip"] = "early_chunk"; return
            if _tts_slow_now():
                seg["sent_skip"] = "slow"; return
            if not _src_subspans(out_dir, seg):
                seg["sent_skip"] = "no_pause"; return
            state_["got"] = _split_sem().acquire(blocking=False)     # at most TTS_SENT_MAX_CONC split slots per chunk at a time
            if not state_["got"]:
                seg["sent_skip"] = "cap"; return
            bak = out_path.with_name(f".{out_path.stem}.legacy{os.getpid()}_{_threading.get_ident()}.wav")
            state_["bak"] = bak                                 # tracked before the copy so a failed/partial copy is cleaned up
            shutil.copyfile(out_path, bak)
            state_["safe"] = False                              # from here on the backup is the only way back, until commit/restore
            if _synth_pieces(seg, voice, out_path, state):
                lf = {k: snap[k] for k in ("ratio", "tempo", "overflow", "overflow_s", "slack_added") if snap[k] is not _MISSING}
                seg["legacy_fit"] = lf
                seg.pop("slack_added", None)                     # the active fields describe the split audio only
                seg.pop("sent_skip", None)
            else:
                seg["sent_skip"] = state.get("skip", "other")
            state_["safe"] = True                               # committed (or nothing was replaced)
        except Exception as e:
            try:
                log.warning("per-sentence TTS seg %s failed (%s); restoring legacy audio", seg.get("id"), e)
            except Exception:
                pass
            try:
                rollback(e)
            except _SplitRestoreFailed:
                raise
            except Exception:
                pass
        except BaseException as e:                              # KeyboardInterrupt / SystemExit: roll back, then propagate
            try:
                rollback(e)
            except Exception:
                pass
            raise
        finally:
            bak = state_["bak"]
            for fn in (lambda: state_["safe"] and bak is not None and bak.exists() and os.remove(bak),
                       lambda: state_["got"] and _split_sem().release(),
                       lambda: (state_["got"] or state.get("made")) and _safe_cleanup(seg.get("id"), tts_dir, state.get("made"))):
                try:
                    fn()
                except Exception:
                    pass

    with ThreadPoolExecutor(max_workers=_env_workers("TTS_WORKERS", max(2, 2 * len(TTS_BASES)))) as ex:
        futures = [ex.submit(synth, seg) for seg in segments if not seg["error"]]
        for fut in as_completed(futures):
            fut.result()  # propagate / log inside

    atomic_write_json(out_dir / "segments.json", segments, indent=2)
    enforce_fallback_budget(segments, "tts")
    try:
        from collections import Counter as _C
        _c = _C(("split" if s.get("sent_pieces") else s.get("sent_skip", "n/a")) for s in segments if not s.get("error"))
        log.info("Stage tts: split outcomes %s", dict(_c))
    except Exception:
        pass
    log.info("Stage tts: done")
    return segments


class TtsStage(StageBase[Segments, Segments]):
    """Synthesise segments with the split and runaway-repair policies."""

    name = "tts"
    policy = StagePolicy(timing_key="tts")

    def run(self, value: Segments, ctx: ChunkContext) -> Segments:
        """Synthesise segments with the split and runaway-repair policies."""
        return from_records(stage_tts(ctx.legacy_args(), ctx.work_dir, to_records(value)))
