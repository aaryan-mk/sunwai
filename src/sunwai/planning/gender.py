"""Stage 6 -- per-speaker gender: ECAPA voice-gender classifier, aggregated into a mean ``p_female``.

``spkgender.json`` caches ``{speaker: {mean_p, probs, total_s, n_turns}}``. The label itself (female / male /
unknown) is assigned later from ``mean_p`` by :func:`assign_gender`, whose band is configurable.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

import numpy as np

from sunwai.planning.envvars import env_float
from sunwai.planning.fileio import atomic_json, json_or_none
from sunwai.planning.speaker_turns import Turn, first_appearance_order, main_speakers, nearest_speaker, speaker_totals
from sunwai.planning.wavio import SAMPLE_RATE, Samples, read_mono16k

log = logging.getLogger("planner")

GenderStats = dict[Any, dict[str, Any]]

_MODEL_ID = "JaesungHuh/voice-gender-classifier"
_MIN_CLIP_S = 1.0
_CLIPS_PER_SPEAKER = 10
_DEFAULT_BAND = (0.6, 0.4)
_warned_bands: set[tuple[float, float]] = set()


def merge_small_speakers(turns: list[Turn], vetted: bool, min_s: float = 5.0) -> list[Turn]:
    """Fold speakers with under ``min_s`` of speech into the temporally nearest main speaker, in place.

    Speakers are then renumbered 0..N-1 by first appearance. When diarization already vetted the speakers by
    identity evidence (``vetted``), nothing is folded: acoustically distinct short speakers stay their own speaker
    and only the renumbering runs.
    """
    totals = speaker_totals(turns)
    if not totals:
        return turns
    big = set(totals) if vetted else main_speakers(totals, min_s)
    big_turns = [t for t in turns if t["speaker"] in big]
    for turn in turns:
        if turn["speaker"] not in big:
            turn["speaker"] = nearest_speaker(big_turns, (turn["start"] + turn["end"]) / 2)
    order = first_appearance_order(turns)
    for turn in turns:
        turn["speaker"] = order.index(turn["speaker"])
    return turns


def normalise_gender_keys(gender: Any) -> Any:
    """JSON round-trips int speaker ids to str keys; return a dict with int keys where possible.

    Kill switch ``GENDER_KEYFIX=0`` keeps the input. Non-numeric keys and non-dict values are kept as they are.
    """
    if os.environ.get("GENDER_KEYFIX", "1") != "1" or not isinstance(gender, dict):
        return gender
    out: dict[Any, Any] = {}
    for key, value in gender.items():
        try:
            out[int(key)] = value
        except (TypeError, ValueError):
            out[key] = value
    return out


def gender_band() -> tuple[float, float]:
    """``(HI, LO)`` from ``GENDER_HI`` / ``GENDER_LO``.

    Invalid values fall back to the validated defaults 0.6 / 0.4 with a warning. An inverted pair (LO > HI) would
    classify every speaker as female, so the whole pair falls back to the defaults.
    """
    hi = env_float("GENDER_HI", _DEFAULT_BAND[0], lo=0.0, hi=1.0)
    lo = env_float("GENDER_LO", _DEFAULT_BAND[1], lo=0.0, hi=1.0)
    if lo > hi:
        if (hi, lo) not in _warned_bands:
            _warned_bands.add((hi, lo))
            log.warning("GENDER_LO=%s > GENDER_HI=%s is inverted; using defaults 0.4/0.6", lo, hi)
        return _DEFAULT_BAND
    return hi, lo


def assign_gender(mean_p: float, total_s: float, n_turns: int) -> str:
    """Label a speaker ``female`` / ``male`` / ``unknown`` from the mean female probability and the amount of speech.

    The 0.6/0.4 band was validated on labelled CV17 + Kathbath (speaker-disjoint DEV): the old 0.9/0.1 abstained on
    15.6% of TEST speakers (unknown means a parity voice, i.e. a coin flip); 0.6/0.4 lowers the expected mismatch.
    """
    hi, lo = gender_band()
    if mean_p != mean_p:  # NaN probability: no evidence
        return "unknown"
    enough = total_s >= 8.0 and (n_turns >= 2 or total_s >= 20.0)
    if mean_p >= hi and enough:
        return "female"
    if mean_p <= lo and enough:
        return "male"
    return "unknown"


def summarise_speaker(turns: Sequence[Turn], probs: list[float]) -> dict[str, Any]:
    """The cached record of one speaker; ``mean_p`` is 0.5 without any usable clip."""
    return {
        "mean_p": float(np.mean(probs)) if probs else 0.5,
        "probs": probs,
        "total_s": sum(t["end"] - t["start"] for t in turns),
        "n_turns": len(turns),
    }


def _female_probability(model: Any, device: Any, clip: Any) -> float:
    import torch

    wave = torch.from_numpy(clip)[None, :].to(device)
    with torch.no_grad():
        logits = model.forward(wave)
        return float(torch.softmax(logits, dim=-1)[0][1])  # index 1 = female


def _classify(samples: Samples, turns: Sequence[Turn]) -> GenderStats:
    """Run the classifier on the longest clips of every speaker."""
    import torch
    from model import ECAPA_gender  # type: ignore[import-not-found]  # the classifier's own package

    model = ECAPA_gender.from_pretrained(_MODEL_ID)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    by_speaker: dict[Any, list[Turn]] = {}
    for turn in turns:
        by_speaker.setdefault(turn["speaker"], []).append(turn)
    stats: GenderStats = {}
    for spk, spk_turns in by_speaker.items():
        longest = sorted(spk_turns, key=lambda t: t["end"] - t["start"], reverse=True)[:_CLIPS_PER_SPEAKER]
        clips = [t for t in longest if t["end"] - t["start"] >= _MIN_CLIP_S]
        probs = [
            _female_probability(model, device, samples[int(t["start"] * SAMPLE_RATE) : int(t["end"] * SAMPLE_RATE)])
            for t in clips
        ]
        stats[spk] = summarise_speaker(spk_turns, probs)
    return stats


def stage_gender(out_dir: str, turns: list[Turn], vetted: bool) -> GenderStats:
    """Return per-speaker gender statistics (cached in ``spkgender.json``); small speakers are merged first."""
    merge_small_speakers(turns, vetted)
    cache_path = os.path.join(out_dir, "spkgender.json")
    cached = json_or_none(cache_path)
    if isinstance(cached, dict):
        return normalise_gender_keys(cached)  # type: ignore[no-any-return]
    stats = _classify(read_mono16k(os.path.join(out_dir, "source16k.wav")), turns)
    atomic_json(cache_path, stats, indent=2)
    return stats
