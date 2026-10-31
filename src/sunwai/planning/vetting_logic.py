"""Identity-evidence speaker vetting: the pure decision logic.

Short speakers are folded into the acoustically most similar main speaker only when speaker-embedding cosine, F0 and
simultaneous speech all agree; two established speakers are merged only when they are clearly one voice (over-split
repair). Everything here is numpy/dict only so it can be unit-tested without models.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from sunwai.planning.envvars import env_flag, env_float
from sunwai.planning.speaker_turns import Turn, main_speakers, nearest_speaker, speaker_totals

Interval = tuple[float, float]
Piece = tuple[float, float, npt.NDArray[np.float32]]
Decision = dict[str, Any]
PairSeconds = dict[tuple[Any, Any], float]

# Calibrated on the 4 demo jobs: same-speaker split null versus different-speaker cosines.
COS_SHORT = 0.60  # centroid cosine needed to fold a < 5 s speaker into a main speaker as the SAME person
COS_MERGE = 0.72  # ... to merge two established speakers (over-split repair); stricter
F0_TOL = 0.25  # max |delta median ln F0| (about 4 semitones) for a merge
SIM_MAX_S = 0.5  # max simultaneous-speech seconds between two labels that are claimed to be one person
MIN_OWN_S = 2.0  # below this much speech a speaker cannot carry an identity: legacy temporal fold
MIN_CLEAN_S = 1.0  # minimum clean (non-overlapped) speech to compute an embedding at all
MIN_TURNS = 2  # over-split repair: each side needs this many turns with clean speech


@dataclass
class VetConstants:
    """Thresholds of the vetting rules (each overridable through an ``SPK_*`` environment variable)."""

    cos_short: float = COS_SHORT
    cos_merge: float = COS_MERGE
    f0_tol: float = F0_TOL
    sim_max_s: float = SIM_MAX_S
    min_own_s: float = MIN_OWN_S
    min_clean_s: float = MIN_CLEAN_S
    min_turns: int = MIN_TURNS
    overlap_on: bool = True
    overlap_merge_gap: float = 0.15
    overlap_min: float = 0.1

    @classmethod
    def from_env(cls) -> VetConstants:
        """Read the thresholds, falling back to the calibrated defaults on invalid values."""
        return cls(
            cos_short=env_float("SPK_COS_SHORT", COS_SHORT, lo=-1.0, hi=1.0),
            cos_merge=env_float("SPK_COS_MERGE", COS_MERGE, lo=-1.0, hi=1.0),
            f0_tol=env_float("SPK_F0_TOL", F0_TOL, lo=0.0, hi=3.0),
            sim_max_s=env_float("SPK_SIM_MAX_S", SIM_MAX_S, lo=0.0),
            min_own_s=env_float("SPK_MIN_OWN_S", MIN_OWN_S, lo=0.0),
            min_clean_s=env_float("SPK_MIN_CLEAN_S", MIN_CLEAN_S, lo=0.2),
            min_turns=int(env_float("SPK_MIN_TURNS", MIN_TURNS, lo=1.0)),
            overlap_on=env_flag("SPK_OVERLAP_SPANS"),
        )


@dataclass
class SpeakerEvidence:
    """Acoustic evidence for one speaker, measured on clean (non-overlapped) speech."""

    pieces: list[Piece] = field(default_factory=list)
    clean_s: float = 0.0
    n_clean_turns: int = 0
    logf0: float | None = None


def subtract_intervals(segments: Sequence[Interval], cuts: Sequence[Interval]) -> list[Interval]:
    """``segments`` minus the union of ``cuts`` (all lists of ``(a, b)``)."""
    out: list[Interval] = []
    for seg in segments:
        current = [seg]
        for a, b in cuts:
            remaining: list[Interval] = []
            for u, v in current:
                if b <= u or a >= v:
                    remaining.append((u, v))
                    continue
                if a > u:
                    remaining.append((u, a))
                if b < v:
                    remaining.append((b, v))
            current = remaining
        out += current
    return out


def overlap_regions(
    full: Sequence[tuple[float, float, str]],
) -> tuple[list[tuple[float, float, frozenset[str]]], PairSeconds]:
    """Sweep the non-exclusive annotation.

    Returns:
        ``(regions, pair_seconds)``: every ``(a, b, labels)`` span where two or more labels are active at once, and
        the simultaneous seconds per label pair.
    """
    events: list[tuple[float, int, str]] = []
    for a, b, label in full:
        if b > a:
            events.append((a, 1, label))
            events.append((b, -1, label))
    events.sort(key=lambda e: (e[0], e[1]))
    active: dict[str, int] = {}
    regions: list[tuple[float, float, frozenset[str]]] = []
    pair: PairSeconds = {}
    last: float | None = None
    for t, delta, label in events:
        if last is not None and t > last and len(active) >= 2:
            labels = frozenset(active)
            regions.append((last, t, labels))
            names = sorted(labels)
            for i, first in enumerate(names):
                for second in names[i + 1 :]:
                    pair[(first, second)] = pair.get((first, second), 0.0) + (t - last)
        active[label] = active.get(label, 0) + delta
        if delta < 0 and active[label] <= 0:
            active.pop(label, None)
        last = t
    return regions, pair


def centroid(pieces: Sequence[Piece]) -> npt.NDArray[np.float32] | None:
    """Duration-weighted, L2-normalised mean embedding, or ``None`` without pieces."""
    if not pieces:
        return None
    weights = np.array([b - a for a, b, _e in pieces], dtype="float64")
    mean = np.average(np.stack([e for _a, _b, e in pieces]), axis=0, weights=weights)
    unit: npt.NDArray[np.float32] = (mean / (np.linalg.norm(mean) + 1e-9)).astype(np.float32)
    return unit


def cosine(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> float:
    """Dot product of two unit vectors."""
    return float(np.dot(a, b))


class _Rules:
    """Shared lookups for the two decision passes."""

    def __init__(self, pair_s: PairSeconds, logf0: dict[Any, float | None], consts: VetConstants) -> None:
        self.pair_s, self.logf0, self.c = pair_s, logf0, consts

    def simultaneous(self, a: Any, b: Any) -> float:
        return self.pair_s.get((a, b), self.pair_s.get((b, a), 0.0))

    def f0_compatible(self, a: Any, b: Any) -> bool:
        fa, fb = self.logf0[a], self.logf0[b]
        return fa is None or fb is None or abs(fa - fb) <= self.c.f0_tol


def _fold_short_speakers(
    turns: Sequence[Turn],
    totals: dict[Any, float],
    big: set[Any],
    cent: dict[Any, Any],
    rules: _Rules,
    mapping: dict[Any, Any],
) -> list[Decision]:
    """Pass 1: fold each short speaker into its most similar main speaker, or into the temporally nearest one."""
    log: list[Decision] = []
    big_turns = [t for t in turns if t["speaker"] in big]
    for k in sorted(totals, key=lambda s: totals[s]):
        if k in big:
            continue
        have = cent[k] is not None and totals[k] >= rules.c.min_own_s
        cands = [(b, cosine(cent[k], cent[b])) for b in big if b != k and cent.get(b) is not None] if have else []
        rec: Decision = {"speaker": k, "total_s": round(totals[k], 2), "rule": "short"}
        if cands:
            b, cs = max(cands, key=lambda x: x[1])
            rec.update(best=b, cos=round(cs, 4))
            same = (
                cs >= rules.c.cos_short and rules.f0_compatible(k, b) and rules.simultaneous(k, b) <= rules.c.sim_max_s
            )
            if same:
                mapping[k] = b
            rec["action"] = "merge_identity" if same else "keep_own"
        else:
            own = [t for t in turns if t["speaker"] == k]
            mid = (own[0]["start"] + own[0]["end"]) / 2 if own else 0.0
            mapping[k] = nearest_speaker(big_turns, mid)  # legacy: fold into the temporally nearest main speaker
            rec["action"] = "fold_no_evidence"
        log.append(rec)
    return log


def _best_merge(
    alive: list[Any],
    pooled: dict[Any, list[Piece]],
    nclean: dict[Any, int],
    members: dict[Any, set[Any]],
    rules: _Rules,
) -> tuple[float, Any, Any, float] | None:
    """Highest-cosine pair of surviving speakers that clears every merge condition."""
    best: tuple[float, Any, Any, float] | None = None
    for i, a in enumerate(alive):
        for b in alive[i + 1 :]:
            ca, cb = centroid(pooled[a]), centroid(pooled[b])
            if ca is None or cb is None or nclean[a] < rules.c.min_turns or nclean[b] < rules.c.min_turns:
                continue
            cs = cosine(ca, cb)
            simult = sum(rules.simultaneous(x, z) for x in members[a] for z in members[b])
            ok = cs >= rules.c.cos_merge and simult <= rules.c.sim_max_s and rules.f0_compatible(a, b)
            if ok and (best is None or cs > best[0]):
                best = (cs, a, b, simult)
    return best


def _repair_oversplit(
    totals: dict[Any, float], evidence: dict[Any, SpeakerEvidence], rules: _Rules, mapping: dict[Any, Any]
) -> list[Decision]:
    """Pass 2: greedily merge surviving speakers that are clearly one voice, pooling centroids after each merge."""
    log: list[Decision] = []
    alive = sorted({mapping[k] for k in mapping})
    pooled = {k: list(evidence[k].pieces) if k in evidence else [] for k in alive}
    members: dict[Any, set[Any]] = {k: {k} for k in alive}
    for k in totals:
        if mapping[k] != k:
            members[mapping[k]].add(k)
    nclean = {k: evidence[k].n_clean_turns if k in evidence else 0 for k in alive}

    def seconds(k: Any) -> float:
        return sum(totals[m] for m in members[k])

    while (best := _best_merge(alive, pooled, nclean, members, rules)) is not None:
        cs, a, b, simult = best
        keep, gone = (a, b) if seconds(a) >= seconds(b) else (b, a)
        log.append(
            {"rule": "oversplit", "keep": keep, "merged": gone, "cos": round(cs, 4),
             "simultaneous_s": round(simult, 3), "action": "merge_identity"}
        )  # fmt: skip
        for k in members[gone]:
            mapping[k] = keep
        members[keep] |= members[gone]
        pooled[keep] += pooled[gone]
        nclean[keep] += nclean[gone]
        alive.remove(gone)
        members.pop(gone)
        pooled.pop(gone)
    return log


def decide_mapping(
    turns: Sequence[Turn],
    evidence: dict[Any, SpeakerEvidence],
    pair_s: PairSeconds,
    consts: VetConstants,
    min_s: float = 5.0,
) -> tuple[dict[Any, Any], list[Decision]]:
    """Decide which speaker labels are the same person.

    1. Short (``< min_s``) speakers with enough clean speech merge into the MOST SIMILAR main speaker iff cosine
       ``>= cos_short``, F0-compatible and not simultaneous with it; otherwise they stay their own speaker. Without
       evidence (``< min_own_s`` of speech or ``< min_clean_s`` clean) they take the legacy temporal fold.
    2. Over-split repair among the remaining speakers: greedy highest-cosine pair with cosine ``>= cos_merge``,
       F0-compatible, simultaneous speech ``<= sim_max_s`` and ``>= min_turns`` clean turns each.

    Returns:
        ``({speaker: merged speaker}, decision log)``.
    """
    totals = speaker_totals(turns)
    if not totals:
        return {}, []
    big = main_speakers(totals, min_s)
    mapping = {k: k for k in totals}
    cent = {k: centroid(evidence[k].pieces) if k in evidence else None for k in totals}
    logf0 = {k: evidence[k].logf0 if k in evidence else None for k in totals}
    rules = _Rules(pair_s, logf0, consts)
    log = _fold_short_speakers(turns, totals, big, cent, rules, mapping)
    log += _repair_oversplit(totals, evidence, rules, mapping)
    return mapping, log


def overlap_spans(
    all_turns: Sequence[tuple[float, float, Any]],
    full: Sequence[tuple[float, float, str]] | None,
    relabel: dict[str, int],
    consts: VetConstants,
) -> list[Decision]:
    """Spans written to ``overlap_spans.json``.

    ``kind="overlap"``: two or more speakers at once (non-exclusive pyannote output). ``kind="short_turn"``: an
    exclusive turn under 0.3 s that the planner drops. Times are seconds on the job's ``source16k`` timeline.
    """
    regions = overlap_regions(full)[0] if full else []
    spans: list[Decision] = []
    for a, b, labels in regions:
        ids = sorted({relabel[label] for label in labels if label in relabel})
        if spans and spans[-1]["kind"] == "overlap" and a - spans[-1]["end"] <= consts.overlap_merge_gap:
            spans[-1]["end"] = b
            spans[-1]["speakers"] = sorted(set(spans[-1]["speakers"]) | set(ids))
        else:
            spans.append({"start": a, "end": b, "kind": "overlap", "speakers": ids})
    spans = [s for s in spans if s["end"] - s["start"] >= consts.overlap_min]
    for a, b, label in all_turns:
        if 0 < (b - a) < 0.3:
            spans.append(
                {
                    "start": a,
                    "end": b,
                    "kind": "short_turn",
                    "speakers": [relabel[str(label)]] if str(label) in relabel else [],
                }
            )
    spans.sort(key=lambda s: (s["start"], s["end"]))
    for s in spans:
        s["start"], s["end"] = round(s["start"], 3), round(s["end"], 3)
    return spans
