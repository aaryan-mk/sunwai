"""The render stages shared by the sequential and pipelined renderers: cluster stream, denoise and restore."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from sunwai.lipsync.denoise import Denoiser
from sunwai.lipsync.models import cuda_sync
from sunwai.lipsync.pasteback import PasteBack, crossfade
from sunwai.lipsync.planner import Alpha, WindowPlanner
from sunwai.lipsync.prepare import Pack, RunPreparer
from sunwai.lipsync.report import ChunkReport, Timings
from sunwai.lipsync.runs import FPS, plan_clusters
from sunwai.lipsync.settings import ChunkConfig

ClusterBatch = tuple[int, int, list[Pack]]


@dataclass
class RenderJob:
    """Everything a renderer needs for one chunk.

    Attributes:
        frames: Original RGB frames ``[N, H, W, 3]``.
        out: Output frames (a copy of ``frames`` that runs are composited into).
        windows: Speech windows ``[start_s, end_s]``.
        config: Render settings.
        job_dir: Chunk directory.
        dub_wav: Dub audio of the chunk.
        scratch_dir: Scratch directory for per-run wavs.
        out_mp4: Where the encoded video is written.
        planner: Plans the face runs of each window.
        preparer: Aligns faces and extracts audio features.
        packs: GPU stages.
        report: Chunk report receiving the rendered runs.
        timings: Stage timings.
        alpha: Cross-fade weights at gate boundaries.
    """

    frames: npt.NDArray[np.uint8]
    out: npt.NDArray[np.uint8]
    windows: Sequence[Sequence[float]]
    config: ChunkConfig
    job_dir: str
    dub_wav: str
    scratch_dir: str
    out_mp4: str
    planner: WindowPlanner
    preparer: RunPreparer
    packs: PackRenderer
    report: ChunkReport
    timings: Timings
    alpha: Alpha

    @property
    def height(self) -> int:
        """Frame height."""
        return int(self.frames.shape[1])

    @property
    def width(self) -> int:
        """Frame width."""
        return int(self.frames.shape[2])


class PackRenderer:
    """GPU stages for packs: denoise (UNet) and restore (paste-back and compositing).

    Args:
        denoiser: Runs the UNet.
        pasteback: Pastes generated faces into frames.
        device: Torch device generated faces are restored on.
        paste_v2: Compute seam colour offsets for the ROI paste-back.
    """

    def __init__(self, denoiser: Denoiser, pasteback: PasteBack, device: Any, paste_v2: bool) -> None:
        self.denoiser = denoiser
        self.pasteback = pasteback
        self.device = device
        self.paste_v2 = paste_v2

    def denoise(self, pack: Pack, config: ChunkConfig, offload: bool) -> None:
        """Generate the pack's faces in place.

        Args:
            pack: The run to denoise; ``faces`` and ``wch`` are dropped afterwards.
            config: Render settings.
            offload: Move the generated faces to CPU (several faces share a cluster: never hold more than one pack's
                faces on the GPU).
        """
        n = len(pack.ids)
        pack.gen = self.denoiser.denoise(
            pack.faces[:n], pack.wch[:n], config.steps, config.gs, config.batch, config.seed
        )
        if self.paste_v2:
            pack.coff = self.pasteback.color_offsets(pack.gen, pack.faces[:n])
        pack.n = n
        pack.faces = None
        pack.wch = []
        if offload:
            pack.gen = [face.cpu() for face in pack.gen]
            pack.offloaded = True

    def restore(self, pack: Pack, out: npt.NDArray[np.uint8], alpha: Alpha) -> None:
        """Paste the pack's generated faces into ``out``, cross-fading at gate boundaries.

        Faces speaking at the same time are composited one after the other onto ``out``.

        Args:
            pack: A denoised pack; ``gen`` is released afterwards.
            out: Output frames, updated in place.
            alpha: Cross-fade weights by ``(face key, frame)``.
        """
        n, ids = pack.n, pack.ids
        assert pack.gen is not None
        generated = [face.to(self.device) for face in pack.gen] if pack.offloaded else pack.gen
        restored = self.pasteback.restore(
            generated, [out[i] for i in ids[:n]], pack.boxes[:n], pack.mats[:n], coffs=pack.coff
        )
        pack.gen = None
        del generated
        for k in range(n):
            weight = alpha.get((pack.key, ids[k]), 1.0)
            out[ids[k]] = crossfade(restored[k], out[ids[k]], weight) if weight < 1.0 else restored[k]


def cluster_stream(job: RenderJob, stop: threading.Event | None = None) -> Iterator[ClusterBatch]:
    """Plan every speech window and prepare its time-disjoint clusters of face runs.

    Args:
        job: The chunk being rendered.
        stop: Event that ends the stream early (pipelined renderer shutting down).

    Yields:
        ``(start_frame, end_frame, packs)`` per cluster, in frame order.

    Raises:
        RuntimeError: Clusters overlap (frames before the previous cluster may already be encoding).
    """
    n_frames = len(job.frames)
    max_frames = int(job.config.max_run_s * FPS) if job.config.max_run_s else 0
    last_end = 0
    for start_s, end_s in job.windows:
        first, last = int(round(start_s * FPS)), min(n_frames, int(round(end_s * FPS)))
        if last <= first:
            continue
        for cluster in plan_clusters(job.planner.plan(first, last), max_frames):
            if stop is not None and stop.is_set():
                return
            if cluster.start < last_end:
                raise RuntimeError(f"pipelined runs overlap ({cluster.start} < {last_end})")
            last_end = cluster.end
            packs: list[Pack] = []
            for run in cluster.runs:
                started = time.time()
                wav_path = os.path.join(job.scratch_dir, f"r_{run.start}_{len(packs)}.wav")
                pack = job.preparer.prepare(
                    job.frames,
                    run.key,
                    run.start,
                    run.end,
                    run.landmarks or {},
                    job.dub_wav,
                    wav_path,
                    job.config.audio_offset_ms,
                )
                cuda_sync()
                job.timings.add("affine_audio", time.time() - started)
                if pack is not None:
                    packs.append(pack)
            if packs:
                yield cluster.start, cluster.end, packs
