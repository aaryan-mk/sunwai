"""Persistent, batched LatentSync engine for the dubbing studio.

Run with cwd = the LatentSync checkout. One :class:`Engine` keeps the models resident on one GPU and renders chunks:

* frames are assembled in memory and encoded once, so there is no shared temp dir that can truncate a chunk, and
  the output duration is verified against the dub before it is published;
* models and the InsightFace session load once, not per chunk or run;
* 16-frame windows are batched through the UNet (:mod:`sunwai.lipsync.denoise`);
* face landmarks are detected once per frame and a frame that fails inside a run reuses the nearest valid ones.

The pieces live in sibling modules: models, faces, planner (gating), prepare, denoise, pasteback, stages and
renderer.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from typing import Any

from sunwai.lipsync.denoise import Denoiser
from sunwai.lipsync.faces import FaceAnalyzer
from sunwai.lipsync.gates import ActiveSpeakerGate, gate_enabled
from sunwai.lipsync.media import decode_frames, duration_s, get_video_info
from sunwai.lipsync.models import LatentSyncModels
from sunwai.lipsync.pasteback import PasteBack
from sunwai.lipsync.planner import Alpha, WindowPlanner, load_gate_audio
from sunwai.lipsync.prepare import RunPreparer
from sunwai.lipsync.renderer import render_pipelined, render_sequential
from sunwai.lipsync.report import AsdReport, ChunkReport, Timings, coverage_warning
from sunwai.lipsync.runs import FPS
from sunwai.lipsync.settings import ChunkConfig, EngineFlags
from sunwai.lipsync.stages import PackRenderer, RenderJob
from sunwai.lipsync.windows import create_segments

MAX_DURATION_DRIFT_S = 0.25


class Engine:
    """LatentSync models resident on one GPU, rendering chunks on request.

    Args:
        device: Torch device string for tensors.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.models = LatentSyncModels(device)
        self.flags = EngineFlags.from_env()
        self.faces = FaceAnalyzer(self.models.image_processor, self.models.resolution)
        self.preparer = RunPreparer(self.faces, self.models, FPS)
        pasteback = PasteBack(self.models.device, self.models.image_processor, self.flags.paste_v2)
        self.packs = PackRenderer(Denoiser(self.models), pasteback, self.models.device, self.flags.paste_v2)
        self.gate = self._load_gate()
        self.load_s = self.models.load_seconds
        print(f"[engine] loaded in {self.load_s:.1f}s", flush=True)

    @staticmethod
    def _load_gate() -> ActiveSpeakerGate | None:
        """Build the active-speaker gate; fail open (no gate) when it cannot be built or is switched off."""
        if not gate_enabled():
            return None
        try:
            return ActiveSpeakerGate.load(device="cuda")
        except Exception as ex:
            print(f"[engine] ASD gate unavailable: {ex}", flush=True)
            return None

    def release_cache(self) -> None:
        """Give cached GPU blocks back after every chunk (workers share GPUs with planner / separation / TTS)."""
        self.models.release_cache()

    def process_chunk(
        self, job_dir: str, lang: str, out_path: str | None = None, report_path: str | None = None, **overrides: Any
    ) -> dict[str, Any]:
        """Lip-sync one dubbed chunk and publish ``dub_<lang>_lipsync.mp4`` atomically.

        Args:
            job_dir: Chunk directory (``src.mp4``, ``dub.wav``, ``segments.json``, ``dub_<lang>.mp4``).
            lang: Target language code.
            out_path: Output video; defaults to ``<job_dir>/dub_<lang>_lipsync.mp4``.
            report_path: Report JSON; defaults to ``<job_dir>/lipsync.json``.
            **overrides: :class:`ChunkConfig` overrides.

        Returns:
            The report that was written to ``report_path``.

        Raises:
            RuntimeError: Encoding failed or the output duration does not match the dub.
        """
        import torch

        config = ChunkConfig.from_overrides(overrides)
        timings = Timings()
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        src = os.path.join(job_dir, "src.mp4")
        out_path = out_path or os.path.join(job_dir, f"dub_{lang}_lipsync.mp4")
        with open(os.path.join(job_dir, "segments.json")) as fh:
            segments = json.load(fh)
        _, width, height = get_video_info(src)
        with timings.measure("decode"):
            frames = decode_frames(src, FPS, width, height)
        total = len(frames) / FPS
        self.models.set_deepcache(config.deepcache)
        windows = create_segments(segments, config.pad, total)
        multiface = self.flags.multiface and bool(config.pipeline)
        asd = AsdReport(self.gate is not None, multiface, self.gate.conf_th if self.gate else None)
        report = ChunkReport(job_dir, lang, config.as_dict(), len(frames), len(windows))
        part_path = out_path + f".{os.getpid()}.part.mp4"
        alpha: Alpha = {}
        scratch = tempfile.mkdtemp(prefix="ls_", dir=os.environ.get("TMPDIR"))
        try:
            planner = WindowPlanner(
                self.faces,
                self.gate,
                load_gate_audio(src, self.gate, asd),
                frames,
                config,
                multiface,
                asd,
                timings,
                alpha,
            )
            job = RenderJob(
                frames=frames,
                out=frames.copy(),
                windows=windows,
                config=config,
                job_dir=job_dir,
                dub_wav=os.path.join(job_dir, "dub.wav"),
                scratch_dir=scratch,
                out_mp4=part_path,
                planner=planner,
                preparer=self.preparer,
                packs=self.packs,
                report=report,
                timings=timings,
                alpha=alpha,
            )
            encoded = (render_pipelined if config.pipeline else render_sequential)(job)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        reference = os.path.join(job_dir, f"dub_{lang}.mp4")
        if not encoded:
            shutil.copyfile(reference, part_path)
        durations = (duration_s(part_path), duration_s(reference))
        if abs(durations[0] - durations[1]) > MAX_DURATION_DRIFT_S:
            os.remove(part_path)
            raise RuntimeError(f"duration mismatch out={durations[0]:.2f} ref={durations[1]:.2f}")
        os.replace(part_path, out_path)
        timings.set("asd", round(asd.secs, 2))
        timings.set("total", time.time() - started)
        peak_gb = round(torch.cuda.max_memory_allocated() / 2**30, 1)
        document = report.to_dict(self._finalize_asd(asd, job_dir), timings, durations, total, peak_gb)
        self._write_report(document, report_path or os.path.join(job_dir, "lipsync.json"))
        self._log_summary(document, timings, config)
        return document

    @staticmethod
    def _finalize_asd(asd: AsdReport, job_dir: str) -> dict[str, Any]:
        """Finalise the gate report and print a coverage warning when little of the speech was lip-synced."""
        finalized = asd.finalize()
        warning = coverage_warning(job_dir, finalized)
        if warning:
            print(warning, flush=True)
        return finalized

    @staticmethod
    def _write_report(document: dict[str, Any], path: str) -> None:
        """Write the report JSON atomically."""
        with open(path + ".tmp", "w") as fh:
            json.dump(document, fh, indent=1)
        os.replace(path + ".tmp", path)

    @staticmethod
    def _log_summary(document: dict[str, Any], timings: Timings, config: ChunkConfig) -> None:
        """Print the one-line chunk summary."""
        t = timings
        print(
            f"[engine] {document['job_dir']} chunk={document['chunk_s']:.1f}s runs={len(document['runs'])} "
            f"synced={document['synced_frames']}f total={t.get('total'):.1f}s "
            f"(dec {t.get('decode'):.1f} det {t.get('detect'):.1f} aff+aud {t.get('affine_audio'):.1f} "
            f"den {t.get('denoise'):.1f} rest {t.get('restore'):.1f} enc {t.get('encode'):.1f}) "
            f"pipeline={config.pipeline}",
            flush=True,
        )
