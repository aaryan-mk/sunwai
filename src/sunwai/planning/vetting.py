"""Identity-evidence speaker vetting (I/O side): embeddings, F0 and the ``spkvet.json`` / ``overlap_spans.json`` files.

The decision rules live in :mod:`sunwai.planning.vetting_logic`; this module measures the evidence they need and
publishes the results.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import numpy as np
import numpy.typing as npt

from sunwai.planning.fileio import atomic_json, remove_file
from sunwai.planning.speaker_turns import Turn, first_appearance_order
from sunwai.planning.vetting_logic import (
    Interval,
    Piece,
    SpeakerEvidence,
    VetConstants,
    decide_mapping,
    overlap_regions,
    overlap_spans,
    subtract_intervals,
)
from sunwai.planning.wavio import Samples

log = logging.getLogger("planner")

_MAX_EMBED_S = 40.0  # audio embedded per speaker
_PIECE_S = 8.0  # embedding window
_F0_SECONDS = 20


def clear_vet_files(out_dir: str) -> None:
    """Drop the derived caches so a retry recomputes them (never touches ``speakers_global.json``)."""
    remove_file(os.path.join(out_dir, "spkvet.json"))
    remove_file(os.path.join(out_dir, "overlap_spans.json"))


def _load_ecapa(savedir: str) -> Any:
    """ECAPA speaker-embedding classifier on CPU."""
    from speechbrain.inference.speaker import EncoderClassifier

    os.makedirs(savedir, exist_ok=True)
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir=savedir, run_opts={"device": "cpu"}
    )


def _embed(classifier: Any, segment: Samples) -> npt.NDArray[np.float32]:
    """Unit-norm ECAPA embedding of one audio segment."""
    import torch

    with torch.no_grad():
        wave = torch.from_numpy(np.ascontiguousarray(segment, dtype="float32"))[None]
        emb = classifier.encode_batch(wave).squeeze().cpu().numpy()
    unit: npt.NDArray[np.float32] = (emb / (np.linalg.norm(emb) + 1e-9)).astype(np.float32)
    return unit


def _log_f0(segment: Samples) -> float | None:
    """Median ln F0 of voiced frames, or ``None`` when there is too little voiced speech."""
    try:
        import librosa

        f0, voiced, _ = librosa.pyin(
            np.asarray(segment, dtype="float32"), fmin=60, fmax=450, sr=16000, frame_length=1024
        )
        f0 = f0[voiced & ~np.isnan(f0)]
        return float(np.median(np.log(f0))) if len(f0) >= 20 else None
    except Exception:
        return None


def _longest_first_budget(intervals: list[Interval], budget: float) -> list[Interval]:
    """Take intervals longest first until ``budget`` seconds are used (the last one is truncated), in time order."""
    chosen: list[Interval] = []
    used = 0.0
    for a, b in sorted(intervals, key=lambda p: (-(p[1] - p[0]), p[0])):
        if used >= budget:
            break
        if used + (b - a) > budget:
            b = a + (budget - used)
        chosen.append((a, b))
        used += b - a
    return sorted(chosen)


def _speaker_evidence(
    classifier: Any,
    samples: Samples,
    sr: int,
    turns: Sequence[Turn],
    spk: Any,
    overlap: Sequence[Interval],
    consts: VetConstants,
) -> SpeakerEvidence:
    """Measure one speaker's clean speech.

    Clean speech is CUMULATIVE: every clean interval counts (also those shorter than ``min_clean_s``). The embeddings
    come from the concatenated clean audio (longest intervals first, at most 40 s, cut into 8 s pieces). A speaker with
    less than ``min_clean_s`` of clean speech gets no pieces (legacy fold).
    """
    base = [(t["start"], t["end"]) for t in turns if t["speaker"] == spk]
    clean = [(a, b) for a, b in subtract_intervals(base, overlap) if b - a >= 0.05]  # fragments < 50 ms are discarded
    total = sum(b - a for a, b in clean)
    own = [t for t in turns if t["speaker"] == spk]
    n_turns = sum(1 for t in own if any(a < t["end"] and b > t["start"] and b - a >= 0.2 for a, b in clean))
    evidence = SpeakerEvidence(clean_s=total, n_clean_turns=n_turns)
    if total < consts.min_clean_s:
        return evidence
    chosen = _longest_first_budget(clean, _MAX_EMBED_S)
    audio = np.concatenate([samples[int(a * sr) : int(b * sr)] for a, b in chosen])
    t0, step = chosen[0][0], int(_PIECE_S * sr)
    pieces: list[Piece] = []
    for i in range(0, len(audio), step):
        segment = audio[i : i + step]
        if len(segment) < int(0.5 * sr) and pieces:
            break
        pieces.append((t0 + i / sr, t0 + (i + len(segment)) / sr, _embed(classifier, segment)))
    evidence.pieces = pieces
    evidence.logf0 = _log_f0(audio[: sr * _F0_SECONDS])
    return evidence


def collect_evidence(
    samples: Samples, sr: int, turns: Sequence[Turn], overlap: Sequence[Interval], savedir: str, consts: VetConstants
) -> dict[Any, SpeakerEvidence]:
    """Per-speaker :class:`SpeakerEvidence` (speakers under ``min_own_s`` also take the legacy fold in the rules)."""
    classifier = _load_ecapa(savedir)
    return {
        spk: _speaker_evidence(classifier, samples, sr, turns, spk, overlap, consts)
        for spk in sorted({t["speaker"] for t in turns})
    }


def _apply_mapping(turns: Sequence[Turn], mapping: dict[Any, Any]) -> tuple[list[Turn], dict[Any, int]]:
    """Relabel turns through ``mapping`` and renumber speakers 0..N-1 by first appearance."""
    merged = [dict(t, speaker=mapping.get(t["speaker"], t["speaker"])) for t in turns]
    order = first_appearance_order(merged)
    final = {k: order.index(k) for k in order}
    return [dict(t, speaker=final[t["speaker"]]) for t in merged], final


def _write_overlap_file(
    out_dir: str,
    duration: float,
    all_turns: Sequence[Any],
    full: Sequence[Any],
    final_of_label: dict[str, int],
    consts: VetConstants,
) -> None:
    spans = overlap_spans(all_turns, full, final_of_label, consts)
    seconds = {
        kind: round(sum(s["end"] - s["start"] for s in spans if s["kind"] == kind), 3)
        for kind in ("overlap", "short_turn")
    }
    doc = {
        "version": 1,
        "source": "pyannote-community-1 speaker_diarization (non-exclusive)",
        "duration": round(duration, 3),
        "overlap_s": seconds["overlap"],
        "short_turn_s": seconds["short_turn"],
        "spans": spans,
    }
    atomic_json(os.path.join(out_dir, "overlap_spans.json"), doc, indent=1)


def vet_speakers(
    out_dir: str,
    samples: Samples,
    sr: int,
    out_turns: list[Turn],
    relabel: dict[str, int],
    all_turns: Sequence[Any],
    full: Sequence[Any] | None,
) -> list[Turn]:
    """Vet the diarized speakers by identity evidence.

    Writes ``overlap_spans.json`` and ``spkvet.json`` and returns the vetted turns (labels 0..N-1 by first
    appearance). ``relabel`` maps ``str(pyannote label)`` to the id used in ``out_turns``.

    Raises:
        RuntimeError: When there is no non-exclusive diarization (the caller falls back to the legacy path).
    """
    if not full:
        raise RuntimeError("no non-exclusive diarization: legacy path")
    consts = VetConstants.from_env()
    regions, pair_by_label = overlap_regions(full)
    pair_s = {(relabel[la], relabel[lb]): s for (la, lb), s in pair_by_label.items() if la in relabel and lb in relabel}
    evidence = collect_evidence(
        samples, sr, out_turns, [(a, b) for a, b, _l in regions], os.path.join(out_dir, ".ecapa"), consts
    )
    mapping, decisions = decide_mapping(out_turns, evidence, pair_s, consts)
    turns, final = _apply_mapping(out_turns, mapping)
    if consts.overlap_on:
        final_of_label = {lab: final[mapping.get(i, i)] for lab, i in relabel.items() if mapping.get(i, i) in final}
        _write_overlap_file(out_dir, len(samples) / float(sr), all_turns, full, final_of_label, consts)
    vet = {
        "vetted": True,
        "version": 1,
        "constants": asdict(consts),
        "mapping": {str(k): v for k, v in mapping.items()},
        "decisions": decisions,
        "evidence": {
            str(k): {"clean_s": round(v.clean_s, 2), "n_clean_turns": v.n_clean_turns, "logf0": v.logf0}
            for k, v in evidence.items()
        },
    }
    atomic_json(os.path.join(out_dir, "spkvet.json"), vet, indent=1)
    log.info(
        "[5] vetting: %d -> %d speakers; %s",
        len(set(mapping)),
        len(final),
        [(d.get("speaker", d.get("merged")), d["action"]) for d in decisions],
    )
    return turns
