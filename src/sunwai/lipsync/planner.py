"""Plans which faces of a speech window are re-animated: detection, tracking and active-speaker gating.

Two strategies exist. The multi-face strategy links every detected face into a track and gates each track on its
own; the single-face strategy (kill switch ``LIP_MULTIFACE=0``, or the fallback when the multi-face gate fails)
follows the one face LatentSync would edit. Both fail open: a gate error leaves the faces ungated.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from sunwai.lipsync.faces import FaceAnalyzer
from sunwai.lipsync.gates import ActiveSpeakerGate, SpeakingTrack
from sunwai.lipsync.media import read_mono_wav16
from sunwai.lipsync.models import cuda_sync
from sunwai.lipsync.report import AsdReport, Timings
from sunwai.lipsync.runs import FPS, FaceRun, Interval, gate_runs, split_into_runs
from sunwai.lipsync.settings import ChunkConfig

Alpha = dict[tuple[tuple[int, int] | None, int], float]


def load_gate_audio(src_video: str, gate: ActiveSpeakerGate | None, asd: AsdReport) -> npt.NDArray[np.int16] | None:
    """Extract the ORIGINAL audio the gate scores against; fail open (None) on any error.

    Args:
        src_video: Source video of the chunk.
        gate: The gate; nothing is read when there is none.
        asd: Report updated with the time spent and any error.

    Returns:
        16 kHz mono samples, or None.
    """
    if gate is None:
        return None
    try:
        started = time.time()
        wav = read_mono_wav16(src_video)
        asd.secs += time.time() - started
        return wav
    except Exception as ex:
        print(f"[engine] ASD_GATE_ERROR fail-open: {ex}", flush=True)
        traceback.print_exc()
        asd.error = str(ex)
        return None


def _constant_mask(mask: npt.NDArray[np.bool_]) -> Callable[[int, int], npt.NDArray[np.bool_]]:
    """A speaking function that ignores its interval and returns a precomputed mask."""
    return lambda _start, _end: mask


class WindowPlanner:
    """Turns speech windows into the face runs that should be lip-synced.

    Args:
        faces: Face detector and aligner.
        gate: Active-speaker gate, or None for no gating.
        gate_wav: Original audio for the gate (None disables gating).
        frames: All chunk frames.
        config: Render settings.
        multiface: Use the multi-face strategy.
        asd: Gate statistics updated while planning.
        timings: Stage timings updated while planning.
        alpha: Cross-fade weights at gate boundaries, keyed by ``(face key, frame)``; filled while planning.
    """

    def __init__(
        self,
        faces: FaceAnalyzer,
        gate: ActiveSpeakerGate | None,
        gate_wav: npt.NDArray[np.int16] | None,
        frames: npt.NDArray[np.uint8],
        config: ChunkConfig,
        multiface: bool,
        asd: AsdReport,
        timings: Timings,
        alpha: Alpha,
    ) -> None:
        self.faces = faces
        self.gate = gate
        self.gate_wav = gate_wav
        self.frames = frames
        self.config = config
        self.multiface = multiface
        self.asd = asd
        self.timings = timings
        self.alpha = alpha

    @property
    def _can_gate(self) -> bool:
        return self.gate is not None and self.gate_wav is not None

    def plan(self, first: int, last: int) -> list[FaceRun]:
        """Plan the window ``[first, last)``.

        Args:
            first: First frame (inclusive).
            last: End frame (exclusive).

        Returns:
            The runs to lip-sync (possibly of several faces, possibly overlapping in time).
        """
        length = last - first
        self.asd.speech_s += length / FPS
        face_mask = np.zeros(length, bool)
        synced_mask = np.zeros(length, bool)
        runs = None
        if self._can_gate and self.multiface:
            runs = self._plan_multiface(first, last, face_mask, synced_mask)
        if runs is None:
            runs = self._plan_single_face(first, last, face_mask, synced_mask)
        self.asd.frames_after += int(synced_mask.sum())
        self.asd.runs_after += len(runs)
        self.asd.face_s += face_mask.sum() / FPS
        self.asd.synced_s += synced_mask.sum() / FPS
        return runs

    def _plan_multiface(
        self, first: int, last: int, face_mask: npt.NDArray[np.bool_], synced_mask: npt.NDArray[np.bool_]
    ) -> list[FaceRun] | None:
        """Gate every face track on its own; None when the gate failed (the caller falls back)."""
        assert self.gate is not None
        try:
            with self.timings.measure("detect"):
                dets = [self.faces.detect_faces(self.frames[f]) for f in range(first, last)]
            started = time.time()
            tracks = self.gate.speaking_tracks(self.frames, self.gate_wav, first, last, dets)  # type: ignore[arg-type]
            cuda_sync()
            self.asd.secs += time.time() - started
            runs = [run for track in tracks for run in self._gate_track(track, first, face_mask, synced_mask)]
            self.asd.tracks += len(tracks)
            return runs
        except Exception as ex:
            print(f"[engine] ASD_GATE_ERROR fail-open (multiface): {ex}", flush=True)
            traceback.print_exc()
            self.asd.error = str(ex)
            face_mask[:] = False
            synced_mask[:] = False
            return None

    def _gate_track(
        self, track: SpeakingTrack, first: int, face_mask: npt.NDArray[np.bool_], synced_mask: npt.NDArray[np.bool_]
    ) -> list[FaceRun]:
        """Turn one scored track into gated runs, recording coverage and cross-fade weights."""
        start, length = int(track.frame[0]), len(track.frame)
        face_mask[start - first : start - first + length] = True
        self.asd.frames_before += length
        key = (first, track.tid)
        gated, weights = gate_runs([(start, start + length)], _constant_mask(track.mask))
        for frame, weight in weights.items():
            self.alpha[(key, frame)] = weight
        for run_start, run_end in gated:
            synced_mask[run_start - first : run_end - first] = True
        return [FaceRun(key, run_start, run_end, track.lm) for run_start, run_end in gated]

    def _plan_single_face(
        self, first: int, last: int, face_mask: npt.NDArray[np.bool_], synced_mask: npt.NDArray[np.bool_]
    ) -> list[FaceRun]:
        """Follow the one face LatentSync edits and gate its runs (when a gate is available)."""
        landmarks: dict[int, Any] = {}
        with self.timings.measure("detect"):
            detected = []
            for f in range(first, last):
                landmarks[f] = self.faces.landmarks3(self.frames[f])
                detected.append(landmarks[f] is not None)
            runs = [
                (first + s, first + e)
                for s, e in split_into_runs(detected, FPS, self.config.min_run, self.config.bridge)
            ]
        self.asd.frames_before += sum(end - start for start, end in runs)
        for start, end in runs:
            face_mask[start - first : end - first] = True
        if self._can_gate and runs:
            runs = self._gate_single_face(runs, landmarks)
        for start, end in runs:
            synced_mask[start - first : end - first] = True
        return [FaceRun(None, start, end, landmarks) for start, end in runs]

    def _gate_single_face(self, runs: list[Interval], landmarks: dict[int, Any]) -> list[Interval]:
        """Gate single-face runs with the nose-point association; fail open."""
        assert self.gate is not None
        gate = self.gate

        def speaking(start: int, end: int) -> Any:
            centers = {f: landmarks[f][2] for f in range(start, end) if landmarks.get(f) is not None}
            return gate.speaking(self.frames, self.gate_wav, start, end, centers)[0]  # type: ignore[arg-type]

        try:
            started = time.time()
            gated, weights = gate_runs(runs, speaking)
            for frame, weight in weights.items():
                self.alpha[(None, frame)] = weight
            cuda_sync()
            self.asd.secs += time.time() - started
            return gated
        except Exception as ex:
            print(f"[engine] ASD_GATE_ERROR fail-open: {ex}", flush=True)
            traceback.print_exc()
            self.asd.error = str(ex)
            return runs
