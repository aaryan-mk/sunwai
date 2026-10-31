"""Chunk renderers: a pipelined one (prepare || denoise || restore + streaming encode) and a sequential one.

The pipelined renderer overlaps CPU preparation, GPU denoising and restoring/encoding on three threads with
bounded queues. The sequential renderer (``pipeline=0``, kill switch) runs the same stages inline and encodes once
at the end. Both produce the same frames.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sunwai.lipsync.media import VideoEncoder
from sunwai.lipsync.models import cuda_sync
from sunwai.lipsync.runs import FPS
from sunwai.lipsync.stages import ClusterBatch, RenderJob, cluster_stream

QUEUE_DEPTH = 2
POLL_S = 0.5


@contextmanager
def side_stream() -> Iterator[None]:
    """Run the enclosed block on its own CUDA stream so stages overlap on the GPU."""
    import torch

    with torch.cuda.stream(torch.cuda.Stream()):
        yield


class Channel:
    """Bounded hand-off queue between stages that gives up when the shared stop event is set.

    Args:
        stop: Event set when any stage failed.
    """

    def __init__(self, stop: threading.Event) -> None:
        self._queue: queue.Queue[Any] = queue.Queue(QUEUE_DEPTH)
        self._stop = stop

    def put(self, item: Any) -> bool:
        """Enqueue ``item``; returns False if the pipeline stopped first."""
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=POLL_S)
                return True
            except queue.Full:
                pass
        return False

    def get(self) -> Any:
        """Dequeue the next item; None means "end of stream" or "pipeline stopped"."""
        while not self._stop.is_set():
            try:
                return self._queue.get(timeout=POLL_S)
            except queue.Empty:
                pass
        return None


def _compose(job: RenderJob, batch: ClusterBatch) -> None:
    """Denoise and restore every pack of a cluster inline, recording the rendered runs."""
    _, _, packs = batch
    started = time.time()
    for pack in packs:
        job.packs.denoise(pack, job.config, offload=False)
    cuda_sync()
    job.timings.add("denoise", time.time() - started)
    started = time.time()
    for pack in packs:
        job.packs.restore(pack, job.out, job.alpha)
        job.report.add_run(pack.f0, pack.f1, pack.n, pack.key, FPS)
    cuda_sync()
    job.timings.add("restore", time.time() - started)


def render_sequential(job: RenderJob) -> bool:
    """Render all clusters inline and encode the result once.

    Args:
        job: The chunk being rendered.

    Returns:
        True when an output was encoded (False when nothing was lip-synced).
    """
    for batch in cluster_stream(job):
        _compose(job, batch)
    if not job.report.runs:
        return False
    with job.timings.measure("encode"):
        VideoEncoder(job.width, job.height, FPS, job.dub_wav, job.config.crf, job.out_mp4).finish(job.out)
    return True


class PipelinedRenderer:
    """Three-stage threaded renderer with a streaming encoder.

    Args:
        job: The chunk being rendered.
    """

    def __init__(self, job: RenderJob) -> None:
        self.job = job
        self.stop = threading.Event()
        self.errors: list[BaseException] = []
        self.to_denoise = Channel(self.stop)
        self.to_restore = Channel(self.stop)
        self.encoder: VideoEncoder | None = None
        self.frames_encoded = 0

    def run(self) -> bool:
        """Render the chunk.

        Returns:
            True when an output was encoded (False when nothing was lip-synced).

        Raises:
            BaseException: The first error raised by any stage; the partial output is removed.
        """
        prepare = threading.Thread(target=self._prepare_stage, daemon=True)
        restore = threading.Thread(target=self._restore_stage, daemon=True)
        prepare.start()
        restore.start()
        try:
            self._denoise_stage()
        finally:
            self.to_restore.put(None)
            prepare.join()
            restore.join()
        if self.errors:
            if self.encoder is not None:
                self.encoder.abort()
            raise self.errors[0]
        started = time.time()
        if self.encoder is not None:
            self.encoder.finish(self.job.out[self.frames_encoded :])
        self.job.timings.set("enc_tail", time.time() - started)
        return self.encoder is not None

    def _fail(self, error: BaseException) -> None:
        self.errors.append(error)
        self.stop.set()

    def _prepare_stage(self) -> None:
        """Plan windows and prepare packs, handing clusters to the denoise stage."""
        try:
            with side_stream():
                for batch in cluster_stream(self.job, self.stop):
                    if not self.to_denoise.put(batch):
                        return
        except BaseException as ex:
            self._fail(ex)
        finally:
            self.to_denoise.put(None)

    def _denoise_stage(self) -> None:
        """Denoise clusters on the calling thread (owns the UNet)."""
        job = self.job
        try:
            while (batch := self.to_denoise.get()) is not None:
                started = time.time()
                for pack in batch[2]:
                    job.packs.denoise(pack, job.config, offload=len(batch[2]) > 1)
                cuda_sync()
                job.timings.add("denoise", time.time() - started)
                if not self.to_restore.put(batch):
                    break
        except BaseException as ex:
            self._fail(ex)

    def _restore_stage(self) -> None:
        """Restore frames, record runs and stream final frames to the encoder."""
        job = self.job
        try:
            with side_stream():
                while (batch := self.to_restore.get()) is not None:
                    end = batch[1]
                    started = time.time()
                    for pack in batch[2]:
                        job.packs.restore(pack, job.out, job.alpha)
                        job.report.add_run(pack.f0, pack.f1, pack.n, pack.key, FPS)
                    cuda_sync()
                    job.timings.add("restore", time.time() - started)
                    self._stream_to_encoder(end)
        except BaseException as ex:
            self._fail(ex)

    def _stream_to_encoder(self, end: int) -> None:
        """Encode frames ``[frames_encoded, end)``, which are final because clusters arrive in frame order."""
        job = self.job
        if self.encoder is None:
            self.encoder = VideoEncoder(job.width, job.height, FPS, job.dub_wav, job.config.crf, job.out_mp4)
        started = time.time()
        if end > self.frames_encoded:
            self.encoder.write(job.out[self.frames_encoded : end])
            self.frames_encoded = end
        job.timings.add("enc_write", time.time() - started)


def render_pipelined(job: RenderJob) -> bool:
    """Render the chunk with the threaded pipeline; see :class:`PipelinedRenderer`."""
    return PipelinedRenderer(job).run()
