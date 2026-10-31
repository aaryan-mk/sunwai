"""Overlap and non-speech event preservation and background ducking."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

from sunwai.pipeline.mixing.levels import _frame_db
from sunwai.pipeline.support.env import _envf


# ---------------------------------------------------------------------------
# Stage 10: Mix + mux
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# (1) keep non-speech vocal events (laughter, breath, singing) that the dub does not cover, (2) bounded smooth
# ducking of the separated bed under the dub's speech, (3) keep the ORIGINAL audio in planner overlap spans.
# Every piece has an env kill switch (parsed robustly: 0/false/off/no -> off) and fails open to the old mix.
# ---------------------------------------------------------------------------
def _preserve_flag(name: str, default: bool = True) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if v in ("0", "false", "off", "no"):
        return False
    if v in ("1", "true", "on", "yes"):
        return True
    return default


MIX_KEEP_EVENTS = _preserve_flag("MIX_KEEP_EVENTS", False)   # default OFF; set 1 to enable


MIX_EVENT_THR = _envf("MIX_EVENT_THR", 0.50, 0.05, 0.99)              # AudioSet event-class probability needed to keep a vocal event


MIX_EVENT_MIN_S = _envf("MIX_EVENT_MIN_S", 0.20, 0.05, 2.0)


MIX_EVENT_MAX_CAND = int(_envf("MIX_EVENT_MAX_CAND", 12, 1, 100))


MIX_DUCK = _preserve_flag("MIX_DUCK", True)


MIX_DUCK_DB = _envf("MIX_DUCK_DB", 3.0, 0.0, 6.0)                     # bed reduction under dub speech (bounded to 6 dB)


MIX_DUCK_ATTACK_MS = _envf("MIX_DUCK_ATTACK_MS", 60.0, 5.0, 500.0)    # gain-fall time constant (look-ahead = 2x so the bed is down BEFORE the onset)


MIX_DUCK_RELEASE_MS = _envf("MIX_DUCK_RELEASE_MS", 400.0, 20.0, 3000.0)


MIX_DUCK_HOLD_MS = _envf("MIX_DUCK_HOLD_MS", 300.0, 0.0, 2000.0)      # speech-activity gaps shorter than this do not release the duck (no pumping)


MIX_KEEP_OVERLAP = _preserve_flag("MIX_KEEP_OVERLAP", True)


MIX_OVERLAP_MAX_CUT = _envf("MIX_OVERLAP_MAX_CUT", 0.5, 0.0, 1.0)     # never cut more than this fraction of one segment's dub speech


_ACTIVITY_HOP = 0.01


_AST = {}


def _dub_activity_frames(t: np.ndarray, sr: int, rel: float = 30.0) -> np.ndarray:
    """bool per 10 ms frame: dub speech present (frame power > p95 - rel dB, floor -70 dB). All-silent input -> all False."""
    db = _frame_db(np.asarray(t, dtype=np.float32), sr, _ACTIVITY_HOP)
    if len(db) == 0:
        return np.zeros(0, dtype=bool)
    return db > max(float(np.percentile(db, 95)) - rel, -70.0)


def _speech_activity(dubbed, N: int, sr: int) -> np.ndarray:
    """global per-frame dub-speech activity (union over placed segments)."""
    hop = int(_ACTIVITY_HOP * sr)
    act = np.zeros(N // hop + 1, dtype=bool)
    for (s, em, t_st) in dubbed:
        a = _dub_activity_frames(t_st[:, 0], sr)
        f0 = s // hop
        n = min(len(a), len(act) - f0)
        if n > 0:
            act[f0:f0 + n] |= a[:n]
    return act


def _duck_gain(act: np.ndarray, depth_db: float, attack_ms: float, release_ms: float, hold_ms: float) -> np.ndarray:
    """Per-frame bed gain (<= 1). act -> hold (forward, after each real offset) -> onset look-ahead (2 x attack, backward only) -> one-pole attack/release.
    Depth is bounded to 6 dB; depth 0 or an all-False activity returns exactly 1.0 everywhere."""
    depth_db = max(0.0, min(6.0, float(depth_db)))
    n = len(act)
    if n == 0 or depth_db <= 0.0 or not act.any():
        return np.ones(n, dtype=np.float64)
    hop_ms = _ACTIVITY_HOP * 1000.0
    k = int(round(hold_ms / hop_ms))
    la = int(round(2.0 * attack_ms / hop_ms))
    a = act.astype(np.int32)
    c = np.concatenate([[0], np.cumsum(a)])
    idx = np.arange(n)
    # hold: forward dilation (bridges gaps <= hold AND keeps the duck for `hold` after the real speech offset)
    lo = np.maximum(0, idx - k)
    a = (c[idx + 1] - c[lo]) > 0
    # look-ahead: BACKWARD dilation only (earlier onsets); real offsets are never moved earlier
    c = np.concatenate([[0], np.cumsum(a.astype(np.int32))])
    hi = np.minimum(n, idx + la + 1)
    a = (c[hi] - c[idx]) > 0
    gmin = 10.0 ** (-depth_db / 20.0)
    tgt = np.where(a, gmin, 1.0)
    ka = math.exp(-_ACTIVITY_HOP / (attack_ms * 1e-3))
    kr = math.exp(-_ACTIVITY_HOP / (release_ms * 1e-3))
    g = np.empty(n, dtype=np.float64)
    prev = 1.0
    for i in range(n):
        c = ka if tgt[i] < prev else kr
        prev = tgt[i] + (prev - tgt[i]) * c
        g[i] = prev
    return g


def _overlap_spans(out_dir: Path, N: int, sr: int):
    """Chunk-time (a, b) sample spans from the planner's <job>/overlap_spans.json (seconds on the job timeline,
    {"spans": [{"start","end","kind"}]}). Missing/unreadable/malformed file -> []."""
    try:
        cands = []
        sm = os.environ.get("SPEAKER_MAP")
        if sm:
            cands.append(Path(os.path.dirname(sm)))
        cands.append(Path(out_dir).resolve().parent.parent)
        doc = None
        for jd in cands:
            p = jd / "overlap_spans.json"
            if p.exists():
                doc = json.load(open(p, encoding="utf-8"))
                break
        if not isinstance(doc, dict):
            return []
        try:
            c0 = float(os.environ.get("CHUNK_START", ""))
        except Exception:
            try:
                c0 = float(json.load(open(Path(out_dir) / "meta.json", encoding="utf-8")).get("start", 0.0))
            except Exception:
                return []                                    # unknown chunk offset: do nothing rather than guess
        kinds = {k.strip() for k in os.environ.get("MIX_OVERLAP_KINDS", "overlap,short_turn").split(",") if k.strip()}
        out = []
        for sp in doc.get("spans", []):
            a, b = float(sp["start"]) - c0, float(sp["end"]) - c0
            if not (math.isfinite(a) and math.isfinite(b)) or b <= a or sp.get("kind", "overlap") not in kinds:
                continue
            a, b = max(0, int(a * sr)), min(N, int(b * sr))
            if b - a > int(0.05 * sr):
                out.append((a, b))
        return out
    except Exception as e:
        log.warning("overlap_spans ignored: %s", e)
        return []


def _guard_overlap(spans, dubbed, sr: int, max_cut: float):
    """accept spans one by one; a span is dropped if the UNION of accepted spans + this one would remove more than max_cut of
    any single dubbed segment's speech frames."""
    hop = int(_ACTIVITY_HOP * sr)
    segs = []
    for (s, em, t_st) in dubbed:
        act = _dub_activity_frames(t_st[:, 0], sr)
        if act.any():
            segs.append((np.arange(len(act)) * hop + s, act, float(act.sum())))
    keep = []
    for (a, b) in spans:
        trial = keep + [(a, b)]
        ok = True
        for (f, act, tot) in segs:
            cov = np.zeros(len(f), dtype=bool)
            for (x, y) in trial:
                cov |= (f >= x) & (f < y)
            if float(np.sum(act & cov)) / tot > max_cut:
                ok = False
                break
        if ok:
            keep.append((a, b))
        else:
            log.info("mix: overlap span %.2f-%.2f s skipped (union would cut > %.0f%% of a dubbed segment)", a / sr, b / sr, 100 * max_cut)
    return keep


def _ast_probs(x16: np.ndarray):
    """AudioSet-AST class probabilities for one 16 kHz mono clip, or None when the model is unavailable (fail-open)."""
    try:
        if "m" not in _AST:
            import glob as _glob
            from transformers import ASTFeatureExtractor, ASTForAudioClassification
            hubs = _glob.glob(os.path.expanduser(os.environ.get("MIX_AST_DIR", "~/.cache/huggingface/hub/models--MIT--ast-finetuned-audioset-10-10-0.4593/snapshots/*")))
            if not hubs:
                raise RuntimeError("no AST snapshot")
            import torch
            torch.set_num_threads(min(4, os.cpu_count() or 1))        # do not starve the co-located model servers
            _AST["fe"] = ASTFeatureExtractor.from_pretrained(hubs[0], local_files_only=True)
            _AST["m"] = ASTForAudioClassification.from_pretrained(hubs[0], local_files_only=True).eval()
            _AST["lab"] = [_AST["m"].config.id2label[i] for i in range(len(_AST["m"].config.id2label))]
        import torch
        inp = _AST["fe"]([x16], sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            p = torch.sigmoid(_AST["m"](**inp).logits)[0].numpy()
        return dict(zip(_AST["lab"], p.tolist()))
    except Exception as e:
        if not _AST.get("warned"):
            log.warning("event detector unavailable (%s); non-speech vocal events not kept", e)
            _AST["warned"] = True
        return None


_EVENT_CLASSES = ("Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle", "Breathing", "Gasp", "Sigh", "Singing", "Humming",
                  "Crying, sobbing", "Cough", "Sneeze", "Groan", "Yawn", "Chant", "Choir")


def _event_spans(orig, vocals, replaced, dub_act, N: int, sr: int):
    """(a, b) sample spans inside the replaced region, NOT covered by dub speech (+120 ms), where the separated vocals are active and
    the AudioSet detector sees a non-speech vocal event on the original mix. Returns (spans, info)."""
    from scipy.ndimage import binary_dilation
    from scipy.signal import resample_poly
    hop = int(_ACTIVITY_HOP * sr)
    nf = N // hop
    if nf < 10:
        return [], {}
    rep_f = replaced[:nf * hop].reshape(nf, hop).mean(axis=1) > 0.5
    da = np.zeros(nf, dtype=bool)
    da[:min(nf, len(dub_act))] = dub_act[:nf]
    free = rep_f & ~binary_dilation(da, iterations=12)
    if not free.any():
        return [], {"cand": 0}
    vm = vocals.mean(axis=1).astype(np.float32)
    vdb = _frame_db(vm, sr, _ACTIVITY_HOP)[:nf]
    vact = vdb > max(float(np.percentile(vdb, 95)) - 30.0, -60.0)
    cand = free & vact
    # runs >= MIX_EVENT_MIN_S, merging gaps <= 100 ms
    cand = binary_dilation(cand, iterations=5) & free
    runs, i = [], 0
    while i < nf:
        if cand[i]:
            j = i
            while j < nf and cand[j]:
                j += 1
            if (j - i) * _ACTIVITY_HOP >= MIX_EVENT_MIN_S:
                runs.append((i, j))
            i = j
        else:
            i += 1
    spans, info = [], {"cand": len(runs), "kept": 0, "probs": []}
    if not runs:
        return [], info
    # classify the separated VOCALS of the candidate run itself (zero-padded to >= 1 s; no neighbouring dub-slot speech, no bed music)
    vm16 = vm
    runs = sorted(runs, key=lambda r: r[0] - r[1])[:MIX_EVENT_MAX_CAND]       # longest first, bounded detector cost per chunk
    for (i, j) in sorted(runs):
        a0, b0 = i * hop, j * hop
        x16 = resample_poly(vm16[a0:b0], 160, 441).astype(np.float32)
        if len(x16) < 16000:
            x16 = np.pad(x16, (0, 16000 - len(x16)))
        pr = _ast_probs(x16)
        if pr is None:
            return [], info
        ev = max((pr.get(k, 0.0), k) for k in _EVENT_CLASSES)
        sp = pr.get("Speech", 0.0)
        info["probs"].append([round(i * _ACTIVITY_HOP, 2), round(j * _ACTIVITY_HOP, 2), ev[1], round(ev[0], 3), round(sp, 3)])
        if ev[0] >= MIX_EVENT_THR and ev[0] >= 0.5 * sp:
            spans.append((a0, b0))
            info["kept"] += 1
    return spans, info


def _span_mask(spans, N: int, sr: int, fade_s: float = 0.02) -> np.ndarray:
    """0..1 mask with short (fade_s) Hann edges for kept events."""
    from scipy.signal.windows import hann
    cf = max(1, int(fade_s * sr))
    mask = np.zeros(N, dtype=np.float32)
    for a, b in spans:
        mask[max(0, a):min(N, b)] = 1.0
    win = hann(2 * cf + 1, sym=False)
    return np.clip(np.convolve(mask, win, mode="same") / win.sum(), 0.0, 1.0).astype(np.float32)
