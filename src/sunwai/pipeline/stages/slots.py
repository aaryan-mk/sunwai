"""Slots stage: bounded speech slots from VAD intervals, with speaker splits and force cuts."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

log = logging.getLogger(__name__)

from sunwai.pipeline import runtime  # noqa: E402

from sunwai.pipeline.context import ChunkContext
from sunwai.pipeline.stage import StagePolicy
from sunwai.pipeline.stages.base import StageBase

from sunwai.media.audio import load_wav_mono
from sunwai.pipeline.support.env import _envf


# ---------------------------------------------------------------------------
# Stage 4: Slots (merge VAD intervals, pad for ASR, label pauses)
# ---------------------------------------------------------------------------
_NEWGEN = {"on": False}      # set by prep_shared while it builds a new-job chunk generation


CROP_HALFGAP = os.environ.get("CROP_HALFGAP", "1").strip().lower() not in ("0", "false", "off", "no")   # kill switch


def stage_slots(out_dir: Path, vad: List[Dict[str, float]]) -> List[Dict[str, Any]]:
    """Build bounded speech slots from VAD intervals (merge, split at speaker changes, force-cut long runs).

    Args:
        out_dir: Chunk directory.
        vad: Speech intervals from the VAD stage.

    Returns:
        Slot records, also written to ``slots.json``.
    """
    slot_path = out_dir / "slots.json"
    if slot_path.exists():
        log.info("Stage slots: cached")
        with open(slot_path, encoding="utf-8") as f:
            return json.load(f)

    total_dur = args_dur if (args_dur := runtime.args_dur) else None

    # Merge VAD intervals into slots
    if not vad:
        with open(slot_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        return []

    merged: List[Dict[str, Any]] = []
    for iv in vad:
        s, e = iv["start"], iv["end"]
        if not merged:
            merged.append({"start": s, "end": e, "pauses": []})
            continue
        last = merged[-1]
        gap = s - last["end"]
        gap_s = max(0.0, gap)
        # classify pause
        if 0.15 <= gap_s <= 0.35:
            ptype = "comma"
        elif 0.35 < gap_s < 0.7:
            ptype = "sentence"
        else:
            ptype = None  # gap <0.15 or >=0.7

        if gap_s < 0.7 and (e - last["start"]) <= 12.0:
            # join
            last["end"] = max(last["end"], e)
            if ptype:
                last["pauses"].append({"gap": gap_s, "type": ptype})
        else:
            # split
            if ptype:
                merged.append({"start": s, "end": e, "pauses": [{"gap": gap_s, "type": ptype}]})
            else:
                merged.append({"start": s, "end": e, "pauses": []})

    # Force-cut runs >25 s at lowest-energy 20 ms frame between 18 and 25 s
    final: List[Dict[str, Any]] = []
    for sl in merged:
        if sl["end"] - sl["start"] <= 25.0:
            final.append(sl)
        else:
            pieces = _force_cut(sl, out_dir)
            final.extend(pieces)

    # Add crop padding (±0.2 s clamped) + assign ids + attach pauses of internal gaps
    for i, sl in enumerate(final):
        sl["id"] = i
        # crop with ±0.2 s
        cs = max(0.0, sl["start"] - 0.2)
        ce = sl["end"] + 0.2
        if CROP_HALFGAP:
            # never pad past the midpoint of the gap to a neighbour (a +-0.2 s pad on touching slots put the
            # neighbour's boundary word in BOTH crops -> repeated in two dubs). Gap <= 0 -> no pad on that side.
            if i > 0:
                cs = max(cs, sl["start"] - max(0.0, (sl["start"] - final[i - 1]["end"]) / 2.0))
            if i < len(final) - 1:
                ce = min(ce, sl["end"] + max(0.0, (final[i + 1]["start"] - sl["end"]) / 2.0))
        # clamp by neighbours / edges: don't overlap neighbour crops
        if i > 0:
            prev_end = final[i - 1]["end"]
            cs = max(cs, min(cs, prev_end - 0.01) if False else cs)  # keep simple: no strict overlap constraint for crop
        if i < len(final) - 1:
            nxt_start = final[i + 1]["start"]
            ce = min(ce, nxt_start + 0.2)
        # keep crop at least a bit larger than slot
        ce = max(ce, sl["end"] + 0.01)
        cs = min(cs, sl["start"] - 0.01)
        cs = max(0.0, cs)
        ce = min(ce, (total_dur or ce))
        if CROP_HALFGAP:        # the +-0.01 minimum pad above must not undo the half-gap on touching slots
            if i > 0:
                cs = max(cs, sl["start"] - max(0.0, (sl["start"] - final[i - 1]["end"]) / 2.0))
            if i < len(final) - 1:
                ce = min(ce, sl["end"] + max(0.0, (final[i + 1]["start"] - sl["end"]) / 2.0))
        sl["crop_start"] = float(cs)
        sl["crop_end"] = float(ce)

    # Ensure sorted by id
    final.sort(key=lambda x: x["start"])
    for i, sl in enumerate(final):
        sl["id"] = i

    # the speaker split runs ONLY while prep_shared is creating a NEW job's chunk generation (_NEWGEN["on"]).
    # Legacy chunks/jobs (and PREP_SHARE=0) never reach this branch: this function is then byte-for-byte the old behaviour.
    if _NEWGEN["on"]:
        final = split_slots_by_speaker(out_dir, final, vad, force=True)
    with open(slot_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    log.info("Stage slots: %d slots", len(final))
    return final


# ---------------------------------------------------------------------------
# split slots at diarization speaker changes (so each slot has ONE speaker, one voice, one start time)
# ---------------------------------------------------------------------------
SPLIT_SPK = os.environ.get("SPLIT_SPK", "1") == "1"   # default ON


SPLIT_MIN_SPK_S = _envf("SPLIT_MIN_SPK_S", 0.6, 0.1, 10.0)     # a speaker needs >= this much speech in the slot to count


SPLIT_MIN_PIECE_S = _envf("SPLIT_MIN_PIECE_S", 0.5, 0.1, 10.0)


def split_slots_by_speaker(out_dir: Path, slots: List[Dict[str, Any]], vad: List[Dict[str, float]],
                           min_spk: float = None, min_piece: float = None, force: bool = False) -> List[Dict[str, Any]]:
    """Split slots at diarization speaker changes. Pure function: returns the new list, writes nothing. Called ONLY from
    stage_slots at first creation of a NEW job's canonical slots (schema 2). Old caches are never re-split/renumbered."""
    smap = os.environ.get("SPEAKER_MAP")
    min_spk = SPLIT_MIN_SPK_S if min_spk is None else min_spk
    min_piece = SPLIT_MIN_PIECE_S if min_piece is None else min_piece
    if force and not (smap and os.path.exists(smap)):
        raise RuntimeError("schema-2 slots must be re-split but SPEAKER_MAP is unavailable")
    if not (smap and os.path.exists(smap) and slots):
        return slots
    try:
        g = json.load(open(smap, encoding="utf-8"))
        off = float(os.environ.get("CHUNK_START", "0"))
        turns = sorted((t["start"] - off, t["end"] - off, int(t["speaker"])) for t in g.get("turns", []))
        ivs = [((v["start"], v["end"]) if isinstance(v, dict) else (v[0], v[1])) for v in vad]
        out: List[Dict[str, Any]] = []
        changed = False
        for sl in slots:
            s, e = sl["start"], sl["end"]
            # who speaks when, inside the slot (clip turns)
            runs = []
            for a, b, spk in turns:
                a, b = max(a, s), min(b, e)
                if b - a <= 0:
                    continue
                if runs and runs[-1][2] == spk:
                    runs[-1][1] = max(runs[-1][1], b)
                else:
                    runs.append([a, b, spk])
            tot: Dict[int, float] = {}
            for a, b, spk in runs:
                tot[spk] = tot.get(spk, 0.0) + (b - a)
            keep = {k for k, v in tot.items() if v >= min_spk}
            runs = [r for r in runs if r[2] in keep]
            merged = []
            for r in runs:                       # re-merge after dropping short intruders
                if merged and merged[-1][2] == r[2]:
                    merged[-1][1] = max(merged[-1][1], r[1])
                else:
                    merged.append(list(r))
            if len(merged) < 2:
                out.append(sl)
                continue
            # boundary between consecutive runs: middle of the silence (or overlap) between them, snapped to the widest
            # VAD gap within +-0.6 s when one exists
            bounds = []
            for r0, r1 in zip(merged, merged[1:]):
                b = (r0[1] + r1[0]) / 2.0
                best = None
                for (a1, b1), (a2, b2) in zip(ivs, ivs[1:]):
                    gap = a2 - b1
                    if gap <= 0.05 or not (s < b1 and a2 < e):
                        continue
                    mid = (a2 + b1) / 2.0
                    if abs(mid - b) <= 0.6 and (best is None or gap > best[0]):
                        best = (gap, mid)
                bounds.append(best[1] if best else b)
            edges = [s] + bounds + [e]
            pieces = []
            for k in range(len(edges) - 1):
                lo, hi = edges[k], edges[k + 1]
                inside = [(max(a, lo), min(b, hi)) for a, b in ivs if min(b, hi) - max(a, lo) > 0.02]
                if inside:
                    ps, pe = inside[0][0], inside[-1][1]
                else:
                    ps, pe = lo, hi
                pauses = []
                for (a1, b1), (a2, b2) in zip(inside, inside[1:]):
                    gp = a2 - b1
                    if 0.15 <= gp <= 0.35:
                        pauses.append({"gap": gp, "type": "comma"})
                    elif 0.35 < gp < 0.7:
                        pauses.append({"gap": gp, "type": "sentence"})
                pieces.append({"start": ps, "end": pe, "pauses": pauses})
            if any(p["end"] - p["start"] < min_piece for p in pieces):
                out.append(sl)                    # a sliver piece would not be dubbable: keep the slot whole
                continue
            for k, p in enumerate(pieces):
                p["crop_start"] = sl["crop_start"] if k == 0 else max(0.0, p["start"] - min(0.2, max(0.0, (p["start"] - pieces[k - 1]["end"]) / 2.0)))
                p["crop_end"] = sl["crop_end"] if k == len(pieces) - 1 else p["end"] + min(0.2, max(0.0, (pieces[k + 1]["start"] - p["end"]) / 2.0))
                p["spk_split"] = True
            log.info("Stage slots: split slot %.2f-%.2f at speaker changes -> %s", s, e,
                     [(round(p["start"], 2), round(p["end"], 2)) for p in pieces])
            out.extend(pieces)
            changed = True
        if not changed:
            return slots
        out.sort(key=lambda x: x["start"])
        for i, sl in enumerate(out):
            sl["id"] = i
        return out
    except Exception as ex:
        if force:
            raise
        log.warning("speaker split skipped: %s", ex)
        return slots


FORCE_CUT_MAX_S = 25.0


def _cut_points(y, s: float, e: float, max_len: float = FORCE_CUT_MAX_S, sr: int = 16000) -> List[float]:
    """Cut times (ascending, strictly inside (s, e)) so that EVERY piece is <= max_len. Each cut is the lowest-energy 20 ms
    frame in [s+0.72*max_len, s+max_len] of the remaining run (18..25 s for the default); a degenerate search window falls
    back to s+max_len-1. Iterative, so the right-hand remainder is cut again."""
    cuts: List[float] = []
    frame_len = int(0.02 * sr)
    cur = s
    while e - cur > max_len + 1e-9:
        lo, hi = cur + 0.72 * max_len, min(cur + max_len, e)
        a = int(lo * sr)
        bnd = min(int(hi * sr), len(y) - frame_len)
        best_pos, best_energy = None, float("inf")
        for p in range(a, bnd, frame_len):
            frame = y[p:p + frame_len]
            energy = float(np.sum(frame * frame))
            if energy < best_energy:
                best_energy, best_pos = energy, p / sr
        if best_pos is None or best_pos <= cur + 1.0 or best_pos >= e - 0.05:
            best_pos = min(cur + max_len - 1.0, e - 0.05)
        cuts.append(best_pos)
        cur = best_pos
    return cuts


def _force_cut(sl: Dict[str, Any], out_dir: Path) -> List[Dict[str, Any]]:
    """Split a slot longer than FORCE_CUT_MAX_S (25 s) into pieces that are ALL <= 25 s, at low-energy frames (recursive)."""
    s, e = sl["start"], sl["end"]
    if e - s <= FORCE_CUT_MAX_S:
        return [sl]
    y = load_wav_mono(out_dir / "src16k.wav", sr=16000)
    edges = [s] + _cut_points(y, s, e) + [e]
    return [{"start": a, "end": b, "pauses": []} for a, b in zip(edges, edges[1:])]


def _zero_slots(d: Path) -> bool:
    """True when <d>/slots.json parses to an EMPTY list (VAD found no speech): an empty crops dir is then complete, not partial."""
    try:
        v = json.load(open(Path(d) / "slots.json", encoding="utf-8"))
        v = v.get("slots", v) if isinstance(v, dict) else v
        return isinstance(v, list) and len(v) == 0
    except Exception:
        return False


class SlotsStage(StageBase[list[dict[str, float]], list[dict[str, Any]]]):
    """Build bounded speech slots from VAD intervals."""

    name = "slots"
    policy = StagePolicy(timing_key="slots")

    def run(self, value: list[dict[str, float]], ctx: ChunkContext) -> list[dict[str, Any]]:
        """Build bounded speech slots from VAD intervals."""
        return stage_slots(ctx.work_dir, value)  # type: ignore[no-any-return]
