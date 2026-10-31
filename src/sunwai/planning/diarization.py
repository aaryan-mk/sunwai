"""Stage 5 -- speaker diarization of the whole file, once: pyannote with an ECAPA clustering fallback.

Outputs ``spkturns.json`` and ``spkbackend.json`` (written last: it is the "stage complete" marker), plus
``spkvet.json`` and ``overlap_spans.json`` when identity vetting ran. A pyannote failure is never silent: the ECAPA
fallback is flagged ``degraded`` in every downstream file.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from sunwai.planning.envvars import env_flag
from sunwai.planning.fileio import atomic_json, json_or_none, load_json, remove_file
from sunwai.planning.speaker_turns import Turn
from sunwai.planning.vetting import clear_vet_files, vet_speakers
from sunwai.planning.wavio import SAMPLE_RATE, Samples, as_array, read_mono16k

log = logging.getLogger("planner")

RawTurn = tuple[float, float, Any]
LabelledSpan = tuple[float, float, str]

PYANNOTE_BACKEND = "pyannote-community-1"
ECAPA_BACKEND = "ecapa-agglomerative"
MIN_TURN_S = 0.3


@dataclass
class PyannoteResult:
    """Outcome of one pyannote run.

    Attributes:
        backend: Backend name, or ``None`` when pyannote failed.
        turns: Exclusive ``(start, end, label)`` turns (``None`` on failure).
        failure: Why pyannote failed (``None`` on success).
        overlap_annotation: The ordinary, overlap-aware annotation as ``(start, end, label)``, if available.
    """

    backend: str | None
    turns: list[RawTurn] | None
    failure: str | None
    overlap_annotation: list[LabelledSpan] | None = None


@dataclass
class DiarizationResult:
    """What stage 5 hands to the later stages.

    Attributes:
        backend: Backend name (``pyannote-community-1`` or ``ecapa-agglomerative``).
        turns: Final turns ``{"start", "end", "speaker"}`` with speakers 0..N-1.
        vetted: True only while the turns were vetted by identity evidence (this run or a cached vetted run).
    """

    backend: str
    turns: list[Turn]
    vetted: bool


def run_pyannote(samples: Samples, sr: int) -> PyannoteResult:
    """Diarize with pyannote; any failure is reported in the result (the caller falls back to ECAPA)."""
    try:
        import torch
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1", token=os.environ.get("HF_TOKEN")
        )
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        out = pipeline({"waveform": torch.from_numpy(as_array(samples))[None], "sample_rate": sr})
        exclusive = next(
            (
                s
                for s in (getattr(out, a, None) for a in ("exclusive_speaker_diarization", "speaker_diarization"))
                if s is not None
            ),
            out,
        )
        turns = [
            (float(seg.start), float(seg.end), label) for seg, _trk, label in exclusive.itertracks(yield_label=True)
        ]
        return PyannoteResult(PYANNOTE_BACKEND, turns, None, _overlap_annotation(out))
    except Exception as exc:  # must fall back on ANY error
        log.warning("[5] pyannote failed -> ECAPA fallback: %s: %s", type(exc).__name__, exc)
        return PyannoteResult(None, None, f"{type(exc).__name__}: {str(exc)[:300]}")


def _overlap_annotation(output: Any) -> list[LabelledSpan] | None:
    """The ordinary (overlap-aware) annotation next to the exclusive one; never fails the diarization."""
    try:
        full = getattr(output, "speaker_diarization", None)
        if full is None:
            return None
        return [(float(s.start), float(s.end), str(label)) for s, _t, label in full.itertracks(yield_label=True)]
    except Exception as exc:
        log.warning("[5] overlap annotation unavailable: %s: %s", type(exc).__name__, exc)
        return None


def cluster_embeddings(feats: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Agglomerative cosine clustering (distance threshold 0.6, i.e. cosine similarity above 0.4)."""
    from sklearn.cluster import AgglomerativeClustering

    if len(feats) == 0:
        return np.array([], dtype=int)
    if len(feats) < 2:  # sklearn cannot cluster one sample -> one speaker
        return np.zeros(len(feats), dtype=int)
    model = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=0.6)
    return model.fit_predict(feats)  # type: ignore[no-any-return]


def windows_to_turns(labels: Any, starts: list[float], win: float) -> list[RawTurn]:
    """Merge consecutive same-label windows into turns."""
    turns: list[RawTurn] = []
    cur_label: Any = None
    cur_start = cur_end = 0.0
    for label, start in zip(labels, starts, strict=False):
        if label != cur_label:
            if cur_label is not None:
                turns.append((cur_start, cur_end, cur_label))
            cur_label, cur_start, cur_end = label, start, start + win
        else:
            cur_end = max(cur_end, start + win)
    if cur_label is not None:
        turns.append((cur_start, cur_end, cur_label))
    return turns


def diarize_ecapa(samples: Samples, sr: int, vad: list[Any], out_dir: str) -> tuple[str, list[RawTurn]]:
    """Fallback diarization: ECAPA embeddings of 1.5 s windows inside VAD intervals, clustered agglomeratively."""
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    savedir = os.path.join(out_dir, ".ecapa")
    os.makedirs(savedir, exist_ok=True)
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=savedir,
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    hop, win = 0.75, 1.5
    feats: list[Any] = []
    starts: list[float] = []
    with torch.no_grad():
        for a, b in vad:
            a, b = float(a), float(b)
            if b - a < 0.4:
                continue
            pos = a
            while pos + win <= b + 1e-3:
                wave = torch.from_numpy(samples[int(pos * sr) : int((pos + win) * sr)])[None, :]
                feats.append(classifier.encode_batch(wave)[0, 0, :].cpu().numpy().astype(np.float32))
                starts.append(pos)
                pos += hop
        matrix = np.vstack(feats) if feats else np.zeros((0, 1), dtype=np.float32)
        if len(matrix):
            matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        labels = cluster_embeddings(matrix)
    return ECAPA_BACKEND, windows_to_turns(labels, starts, win)


def normalise_turns(raw: list[RawTurn]) -> tuple[list[Turn], dict[str, int]]:
    """Drop turns under 0.3 s, sort, and relabel speakers 0..N-1 by first appearance.

    Returns:
        ``(turns, relabel)`` where ``relabel`` maps ``str(original label)`` to the new id.
    """
    kept = sorted((t for t in raw if (t[1] - t[0]) >= MIN_TURN_S), key=lambda t: (t[0], t[1]))
    relabel: dict[str, int] = {}
    turns: list[Turn] = []
    for start, end, label in kept:
        key = str(label)
        if key not in relabel:
            relabel[key] = len(relabel)
        turns.append({"start": round(start, 3), "end": round(end, 3), "speaker": relabel[key]})
    return turns, relabel


def _vet_or_fall_back(
    out_dir: str,
    samples: Samples,
    turns: list[Turn],
    relabel: dict[str, int],
    raw: list[RawTurn],
    full: list[LabelledSpan],
) -> tuple[list[Turn], bool]:
    """Identity-evidence vetting; on any failure keep the unvetted turns (legacy speaker handling)."""
    try:
        return vet_speakers(out_dir, samples, SAMPLE_RATE, turns, relabel, raw, full), True
    except Exception as exc:
        log.warning("[5] vetting failed -> legacy speaker handling: %s: %s", type(exc).__name__, exc)
        clear_vet_files(out_dir)
        return turns, False


def _cached(out_dir: str) -> DiarizationResult | None:
    backend = json_or_none(os.path.join(out_dir, "spkbackend.json"))
    turns = json_or_none(os.path.join(out_dir, "spkturns.json"))
    if not (isinstance(backend, dict) and "backend" in backend and isinstance(turns, list)):
        return None
    log.info("[5] cached")
    vet = json_or_none(os.path.join(out_dir, "spkvet.json"))  # only jobs vetted at planning time stay vetted
    return DiarizationResult(backend["backend"], turns, isinstance(vet, dict) and vet.get("vetted") is True)


def stage_diarize(out_dir: str) -> DiarizationResult:
    """Diarize ``source16k.wav`` (or return the cached result)."""
    cached = _cached(out_dir)
    if cached:
        return cached
    vad = load_json(os.path.join(out_dir, "vad.json"))
    remove_file(
        os.path.join(out_dir, "spkbackend.json")
    )  # the completion marker must not outlive the data it describes
    clear_vet_files(out_dir)
    samples = read_mono16k(os.path.join(out_dir, "source16k.wav"))

    pyannote = run_pyannote(samples, SAMPLE_RATE)
    degraded, reason = False, None
    if pyannote.backend and pyannote.turns is not None:
        backend, raw = pyannote.backend, pyannote.turns
    else:
        # the weaker ECAPA clustering must never be silent: flagged degraded in spkbackend.json,
        # speakers_global.json and plan.json
        degraded, reason = True, f"pyannote failed: {pyannote.failure}"
        log.warning("[5] DEGRADED diarization: %s -> ECAPA agglomerative fallback", reason)
        backend, raw = diarize_ecapa(samples, SAMPLE_RATE, vad, out_dir)

    turns, relabel = normalise_turns(raw)
    vetted = False
    if not degraded and pyannote.overlap_annotation is not None and env_flag("SPK_EVIDENCE"):
        turns, vetted = _vet_or_fall_back(out_dir, samples, turns, relabel, raw, pyannote.overlap_annotation)

    atomic_json(os.path.join(out_dir, "spkturns.json"), turns, indent=2)
    # written LAST: it is the "stage complete" marker
    atomic_json(
        os.path.join(out_dir, "spkbackend.json"),
        {"backend": backend, "degraded": degraded, "degraded_reason": reason},
        indent=2,
    )
    log.info("[5] diarized (%s) -> %d turns, %d speakers", backend, len(turns), len({t["speaker"] for t in turns}))
    return DiarizationResult(backend, turns, vetted)
