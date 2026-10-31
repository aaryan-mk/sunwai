"""Language-independent chunk preparation shared across target languages, with generation markers."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict


log = logging.getLogger(__name__)

from sunwai.media.audio import _wav_ok
from sunwai.pipeline.stages.asr import _reset_downstream, stage_asr
from sunwai.pipeline.stages.audio import stage_audio
from sunwai.pipeline.stages.fetch import stage_fetch
from sunwai.pipeline.stages.slots import SPLIT_MIN_PIECE_S, SPLIT_MIN_SPK_S, SPLIT_SPK, _NEWGEN, _zero_slots, stage_slots
from sunwai.pipeline.stages.speakers import stage_speakers
from sunwai.pipeline.stages.vad import stage_vad
from sunwai.pipeline.support.guards import media_ok
from sunwai.pipeline.support.storage import _atomic_copy, _sha_file, atomic_write_json, save_timing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _early_chunk(out_dir: Path) -> bool:
    """Latency-critical first chunks (index < REFIT_MIN_CHUNK, default 2) skip the serial Gemini refit + re-synthesis;
    an overflowing segment is then handled by the tempo cap + truncation-with-fade already in stage_tts."""
    import re
    try:
        m = re.fullmatch(r"c(\d+)", Path(out_dir).name)
        return bool(m) and int(m.group(1)) < int(os.environ.get("REFIT_MIN_CHUNK", "0"))
    except Exception:
        return False


def chunk_lookahead_gate(out_dir: Path):
    """Playback needs chunks in order, but the server starts every chunk of every language at once, so chunk n+1..n+3
    steal TTS/MT/CPU from chunk n and ALL chunks finish late. Keep a window of CHUNK_LOOKAHEAD chunks per language:
    chunk k starts only when chunk k-L of the same language is done/error (or after CHUNK_GATE_MAX_WAIT=360 s (bounds a stale running state)). L=0 disables."""
    import re
    try:
        L = int(os.environ.get("CHUNK_LOOKAHEAD", "3"))
        m = re.fullmatch(r"c(\d+)", out_dir.name)
        if L <= 0 or not m or int(m.group(1)) < L:
            return
        k = int(m.group(1))
        prev = out_dir.parent / f"c{k - L:04d}" / "status.json"
        deadline = time.time() + float(os.environ.get("CHUNK_GATE_MAX_WAIT", "360"))
        t0 = time.time()
        while time.time() < deadline:
            try:
                st = json.load(open(prev, encoding="utf-8")).get("state")
            except Exception:
                st = None
            if st in ("done", "error", "failed"):
                break
            time.sleep(1.0)
        log.info("chunk gate: waited %.1fs for %s (state=%s)", time.time() - t0, prev.parent.name, st)
        save_timing(out_dir, "gate_wait", time.time() - t0)
    except Exception as e:  # never block the pipeline on a gate bug
        log.warning("chunk gate failed (%s); continuing", e)


def _prep_stages(args, out_dir: Path):
    """language-independent front half: fetch -> audio -> vad -> slots -> speakers -> asr (source-language transcript)."""
    t0 = time.time()
    meta = stage_fetch(args, out_dir)
    save_timing(out_dir, "fetch", time.time() - t0)

    t0 = time.time(); stage_audio(out_dir); save_timing(out_dir, "audio", time.time() - t0)
    t0 = time.time(); vad = stage_vad(out_dir); save_timing(out_dir, "vad", time.time() - t0)
    t0 = time.time(); slots = stage_slots(out_dir, vad); save_timing(out_dir, "slots", time.time() - t0)
    t0 = time.time(); _ = stage_speakers(out_dir, slots); save_timing(out_dir, "speakers", time.time() - t0)
    t0 = time.time(); segments = stage_asr(args, out_dir, slots); save_timing(out_dir, "asr", time.time() - t0)
    return segments


_PREP_FILES = ["meta.json", "src.mp4", "src16k.wav", "src44k.wav", "vad.json", "slots.json", "speakers.json",
               "lid.json", "segments.json", "crops", ".lid"]


def _prep_file_ok(path: Path) -> bool:
    """A prep artefact is usable only if it is complete: wav decodes, mp4 probes, json parses, dirs non-empty."""
    try:
        if not path.exists():
            return False
        if path.is_dir():
            # a chunk with NO speech has zero slots and therefore an empty crops dir; that is a complete, valid prep (the dub keeps the original audio)
            return path.name != "crops" or any(path.iterdir()) or _zero_slots(path.parent)
        suf = path.suffix.lower()
        if suf == ".wav":
            return _wav_ok(path)
        if suf == ".mp4":
            return media_ok(path, need_video=True, need_audio=True, min_dur=0.05)
        if suf == ".json":
            json.load(open(path, encoding="utf-8"))
            return True
        return path.stat().st_size >= 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# NEW-JOB chunk generations (speaker-split slots). Design: the split decision is made ONCE, when a chunk's shared
# prep (<job>/_prep/cNNNN) is first created under its flock. Everything that existed before (no slot_gen.json) is LEGACY and
# runs the old prep_shared body unchanged (no checks, no markers). A new generation = vad.json + slots.json + speakers.json
# + slot_gen.json (marker, written LAST = the commit record). The marker fingerprints the inputs (SPEAKER_MAP content,
# CHUNK_START, thresholds, VAD content) and the produced slots/speakers. Language dirs take the WHOLE generation from _prep
# (replacing any local copy that is not byte-identical) and verify it; every failure here is a hard error.
# ---------------------------------------------------------------------------
GEN_FILE = "slot_gen.json"


_GEN_CORE = ["vad.json", "slots.json", "speakers.json"]


_SLOT_DEP = ["slots.json", "speakers.json", "segments.json", "segments.failed.json", "slot_gen.json", "fit.json",
             "lipsync.json", "crops", "tts"]


_NEWGEN_PENDING = ".newgen_creating"


def _gen_inputs(vad_path) -> Dict[str, Any]:
    return {"smap": _sha_file(os.environ.get("SPEAKER_MAP", "")), "chunk_start": str(os.environ.get("CHUNK_START", "0")),
            "min_spk": SPLIT_MIN_SPK_S, "min_piece": SPLIT_MIN_PIECE_S, "vad": _sha_file(vad_path)}


def _gen_marker(d: Path) -> Dict[str, Any]:
    return {"gen": 1, "inputs": _gen_inputs(d / "vad.json"), "slots": _sha_file(d / "slots.json"),
            "speakers": _sha_file(d / "speakers.json")}


def _gen_read(d: Path):
    p = d / GEN_FILE
    if not p.exists():
        return None
    m = json.load(open(p, encoding="utf-8"))         # corrupt marker -> exception (hard)
    if not isinstance(m, dict) or m.get("gen") != 1:
        raise RuntimeError(f"{p}: unsupported slot generation marker")
    return m


def _gen_verify(d: Path) -> Dict[str, Any]:
    """The marker in d must match the bytes of d's vad/slots/speakers. Raises RuntimeError on any mismatch."""
    m = _gen_read(d)
    if m is None:
        raise RuntimeError(f"{d}: slot generation marker missing")
    for k, f in (("slots", "slots.json"), ("speakers", "speakers.json")):
        if m.get(k) != _sha_file(d / f):
            raise RuntimeError(f"{d}: {f} does not match its generation fingerprint (marker {m.get(k)} vs {_sha_file(d / f)})")
    if m["inputs"].get("vad") != _sha_file(d / "vad.json"):
        raise RuntimeError(f"{d}: vad.json does not match its generation fingerprint")
    return m


def _gen_inputs_changed(m: Dict[str, Any]) -> bool:
    cur = _gen_inputs(Path("/nonexistent"))
    return any(m["inputs"].get(k) != cur[k] for k in ("smap", "chunk_start", "min_spk", "min_piece"))


def _purge_slot_dep(d: Path):
    import shutil
    for n in _SLOT_DEP:
        p = d / n
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists() or p.is_symlink():
            p.unlink()
    for p in list(d.glob("dub*")) + list(d.glob(".dubtmp*")):
        try:
            p.unlink()
        except OSError:
            pass


def _chunk_mode(job_dir: Path, out_dir: Path, sh: Path) -> str:
    """'new' | 'legacy', decided under the chunk's flock. Anything that predates the marker is legacy."""
    if (sh / GEN_FILE).exists() or (sh / _NEWGEN_PENDING).exists():
        return "new"
    if not SPLIT_SPK:
        return "legacy"
    sm = os.environ.get("SPEAKER_MAP")
    if not (sm and os.path.exists(sm)):
        return "legacy"
    if (sh / "PREP_DONE").exists():
        return "legacy"
    for d in job_dir.glob(f"*/{out_dir.name}"):
        if any((d / n).exists() for n in ("slots.json", "segments.json", "speakers.json")):
            return "legacy"
    return "new"


def _publish_files(src: Path, dst: Path, names):
    for n in names:
        if (src / n).exists():
            if not _prep_file_ok(src / n):
                raise RuntimeError(f"refusing to publish partial prep file {n}")
            _atomic_copy(src / n, dst / n)


def _prep_shared_new(args, out_dir: Path, job_dir: Path, sh: Path):
    done = sh / "PREP_DONE"
    m = _gen_read(sh)
    if m is not None and _gen_inputs_changed(m):
        log.error("slot generation inputs changed; re-plan required (%s): SPEAKER_MAP/CHUNK_START/thresholds differ from the "
                  "fingerprint in %s. Nothing was modified.", sh, GEN_FILE)
        raise RuntimeError("slot generation inputs changed; re-plan required")
    if m is not None:
        # fence: core files must not be newer than the commit record (an old-binary writer after the commit)
        mt = (sh / GEN_FILE).stat().st_mtime
        for n in _GEN_CORE:
            if (sh / n).stat().st_mtime > mt + 1.0:
                raise RuntimeError(f"{sh / n} was modified after its slot generation was committed")
    if m is None:
        # ---- create the generation (first language to hold the flock) ----
        _purge_slot_dep(out_dir)
        (sh / _NEWGEN_PENDING).touch()
        _NEWGEN["on"] = True
        try:
            segments = _prep_stages(args, out_dir)
        except BaseException:
            _purge_slot_dep(out_dir)         # leave nothing that a retry could mistake for a legacy chunk
            raise
        finally:
            _NEWGEN["on"] = False
        marker = _gen_marker(out_dir)
        atomic_write_json(out_dir / GEN_FILE, marker)        # failure -> exception -> chunk fails (no swallow)
        _publish_files(out_dir, sh, _GEN_CORE)
        atomic_write_json(sh / GEN_FILE, marker)             # COMMIT RECORD, written last
        _gen_verify(sh)
        (sh / _NEWGEN_PENDING).unlink()
        log.info("Prep shared: new slot generation committed (%d slots) at %s", len(segments), sh)
        if not any(sg.get("fallback_reason") == "asr" for sg in segments):
            _publish_files(out_dir, sh, [n for n in _PREP_FILES if n not in _GEN_CORE])
            done.touch()
        return segments
    # ---- consume an existing generation ----
    m = _gen_verify(sh)
    lm = None
    try:
        lm = _gen_read(out_dir)
    except Exception:
        lm = None
    same = lm == m and all(_sha_file(out_dir / f) == _sha_file(sh / f) for f in _GEN_CORE)
    if not same:
        log.warning("Prep shared: replacing local slot-dependent artifacts of %s with the shared generation", out_dir)
        _purge_slot_dep(out_dir)
    for n in _GEN_CORE:
        if not same or not (out_dir / n).exists():
            _atomic_copy(sh / n, out_dir / n)
    _atomic_copy(sh / GEN_FILE, out_dir / GEN_FILE)
    _gen_verify(out_dir)
    if done.exists():
        for n in _PREP_FILES:
            if n in _GEN_CORE or n == GEN_FILE:
                continue
            src, dst = sh / n, out_dir / n
            if src.exists() and not (dst.exists() and _prep_file_ok(dst)):
                if not _prep_file_ok(src):
                    raise RuntimeError(f"shared prep file {n} is stale/partial")
                _atomic_copy(src, dst)
        segments = _reset_downstream(json.load(open(out_dir / "segments.json", encoding="utf-8")))
    else:
        segments = _prep_stages(args, out_dir)          # creator had ASR failures: retry on the copied generation
        if not any(sg.get("fallback_reason") == "asr" for sg in segments):
            _publish_files(out_dir, sh, [n for n in _PREP_FILES if n not in _GEN_CORE])
            done.touch()
    sl = json.load(open(out_dir / "slots.json", encoding="utf-8"))
    if [x["id"] for x in sl] != [x["id"] for x in segments]:
        raise RuntimeError(f"{out_dir}: segments ids differ from the slot generation")
    return segments


def prep_shared(args, out_dir: Path):
    """Compute the language-independent front half ONCE per (job, chunk) and share it across target languages.
    Layout <job>/<lang>/cNNNN: the first language to take the flock computes and publishes to <job>/_prep/cNNNN,
    the others copy it (PREP_DONE is written last). Any failure falls back to computing locally."""
    import fcntl, shutil
    job_dir = out_dir.parent.parent
    if os.environ.get("PREP_SHARE", "1") != "1" or not (job_dir / "plan.json").exists() or not out_dir.name.startswith("c"):
        return _prep_stages(args, out_dir)
    sh = job_dir / "_prep" / out_dir.name
    sh.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(sh / ".lock", "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        if _chunk_mode(job_dir, out_dir, sh) == "new":
            return _prep_shared_new(args, out_dir, job_dir, sh)          # new-job generation (hard errors)
        done = sh / "PREP_DONE"                                           # ---- LEGACY path: unchanged below ----
        if done.exists():
            try:
                for n in _PREP_FILES:
                    src, dst = sh / n, out_dir / n
                    if not src.exists():
                        continue
                    if dst.exists() and _prep_file_ok(dst):
                        continue            # complete local copy
                    if not _prep_file_ok(src):
                        raise RuntimeError(f"shared prep file {n} is stale/partial")
                    _atomic_copy(src, dst)
                with open(out_dir / "segments.json", encoding="utf-8") as f:
                    segments = _reset_downstream(json.load(f))
                log.info("Prep shared: copied from %s in %.1fs", sh, time.time() - t0)
                save_timing(out_dir, "prep_shared_copy", time.time() - t0)
                return segments
            except Exception as e:
                log.warning("shared prep copy failed (%s); recomputing locally and republishing", e)
                try:
                    done.unlink()
                except OSError:
                    pass
                for n in _PREP_FILES:
                    if (out_dir / n).exists() and not _prep_file_ok(out_dir / n):
                        shutil.rmtree(out_dir / n) if (out_dir / n).is_dir() else (out_dir / n).unlink()
                segments = _prep_stages(args, out_dir)
                if any(sg.get("fallback_reason") == "asr" for sg in segments):
                    return segments
                try:
                    for n in _PREP_FILES:
                        if (out_dir / n).exists() and _prep_file_ok(out_dir / n):
                            _atomic_copy(out_dir / n, sh / n)
                    done.touch()
                except Exception as e2:
                    log.warning("could not republish shared prep (%s)", e2)
                return segments
        segments = _prep_stages(args, out_dir)
        if any(sg.get("fallback_reason") == "asr" for sg in segments):
            log.warning("Prep shared: NOT publishing (some ASR slots failed; each language retries them)")
            return segments
        try:
            for n in _PREP_FILES:
                src, dst = out_dir / n, sh / n
                if not src.exists():
                    continue
                if not _prep_file_ok(src):
                    raise RuntimeError(f"refusing to publish partial prep file {n}")
                _atomic_copy(src, dst)
            done.touch()
            log.info("Prep shared: published to %s", sh)
        except Exception as e:
            log.warning("could not publish shared prep (%s)", e)
        return segments
