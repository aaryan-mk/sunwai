"""GPU htdemucs separation service (one model resident, requests serialised).

``POST /separate {"in_wav": ".../src44k.wav", "out_dir": "..."}`` writes ``out_dir/vocals.wav`` and ``bed.wav``
(the same math as ``dub_pipeline.stage_separate``). Paths must lie under ``SEP_ROOT``. The model is loaded when the
app starts.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import demucs.apply as demucs_apply
import demucs.pretrained as demucs_pretrained
import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from sunwai.mirrors.body_limit import BodyLimitMiddleware
from sunwai.mirrors.separator_config import SeparatorConfig

log = logging.getLogger("sep_server")
logging.basicConfig(level=logging.INFO)
SAMPLE_RATE = 44100


class SeparateRequest(BaseModel):
    """Separation request."""

    in_wav: str
    out_dir: str


class Abandoned(Exception):
    """The client went away before the work started."""


def _check(abandoned: threading.Event) -> None:
    if abandoned.is_set():
        raise Abandoned()


def _write_stems(out_dir: str, stems: dict[str, Any]) -> None:
    """Write each stem next to its destination, then publish them all with ``os.replace``."""
    os.makedirs(out_dir, exist_ok=True)
    pending: list[tuple[str, str]] = []
    unique = f"{os.getpid()}.{threading.get_ident()}.{time.time_ns()}"
    try:
        for name, audio in stems.items():
            tmp = f"{out_dir}/.{name}.{unique}.tmp.wav"
            pending.append((tmp, f"{out_dir}/{name}.wav"))
            sf.write(tmp, audio.T, SAMPLE_RATE)
        for tmp, dst in pending:
            os.replace(tmp, dst)
    finally:
        for tmp, _ in pending:
            try:
                os.remove(tmp)
            except OSError:
                pass


class Separator:
    """The resident htdemucs model with serialised access and bounded queueing."""

    def __init__(self, config: SeparatorConfig) -> None:
        """Create the service (the model loads in :meth:`load`)."""
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Any = None
        self.pending = 0
        self._model_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._decode_slots = threading.Semaphore(config.decode_parallel)

    def load(self) -> None:
        """Load htdemucs onto the device."""
        self.model = demucs_pretrained.get_model("htdemucs").to(self.device).eval()

    def try_enter(self) -> None:
        """Take a pending slot or raise 503 when the queue is full."""
        with self._pending_lock:
            if self.pending >= self.config.max_pending:
                raise HTTPException(
                    503,
                    f"separation service is overloaded ({self.config.max_pending} queued); retry shortly",
                    headers={"Retry-After": "5"},
                )
            self.pending += 1

    def leave(self) -> None:
        """Free a pending slot."""
        with self._pending_lock:
            self.pending -= 1

    def _decode(self, in_wav: str, abandoned: threading.Event) -> Any:
        try:
            size = os.path.getsize(in_wav)
        except OSError:
            raise HTTPException(404, "input wav not found") from None
        if size > self.config.max_bytes:
            raise HTTPException(413, f"input wav too large ({size} bytes; limit {self.config.max_bytes})")
        _check(abandoned)
        with self._decode_slots:
            _check(abandoned)
            audio, _ = librosa.load(in_wav, sr=SAMPLE_RATE, mono=False, dtype="float32")
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        return torch.from_numpy(audio).unsqueeze(0)

    def _acquire_model(self, abandoned: threading.Event) -> None:
        """Wait for the model lock, polling so an abandoned or expired request leaves the queue promptly."""
        deadline = time.time() + self.config.queue_timeout
        while not self._model_lock.acquire(timeout=0.25):
            _check(abandoned)
            if time.time() > deadline:
                raise HTTPException(503, "separation queue wait exceeded the deadline", headers={"Retry-After": "5"})

    def _separate_stems(self, signal: Any) -> dict[str, Any]:
        with torch.no_grad():
            out = demucs_apply.apply_model(self.model, signal, split=True, overlap=0.25, device=self.device)
            stems = out[0].cpu().numpy()
        names = list(self.model.sources)
        vocals_idx = names.index("vocals") if "vocals" in names else len(names) - 1
        return {"vocals": stems[vocals_idx], "bed": stems.sum(axis=0) - stems[vocals_idx]}

    def work(self, request: SeparateRequest, abandoned: threading.Event, started: float) -> dict[str, Any]:
        """Run one separation in a worker thread; ``abandoned`` is set by the async watcher when the client left."""
        signal = self._decode(request.in_wav, abandoned)
        self._acquire_model(abandoned)
        try:
            _check(abandoned)
            stems = self._separate_stems(signal)
            # Publication stays INSIDE the lock: vocals+bed of two requests can never interleave.
            _write_stems(request.out_dir, stems)
        finally:
            self._model_lock.release()
        log.info("separated %s in %.1fs", request.in_wav, time.time() - started)
        return {"ok": True, "secs": time.time() - started}


async def _watch_disconnect(request: Request, abandoned: threading.Event) -> None:
    """Set ``abandoned`` when the client disconnects."""
    while True:
        await asyncio.sleep(0.25)
        try:
            if await request.is_disconnected():
                abandoned.set()
                return
        except Exception:
            return


def create_app(separator: Separator) -> FastAPI:
    """Build the FastAPI application around ``separator`` (the model loads on startup)."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        separator.load()
        yield

    api = FastAPI(lifespan=lifespan)
    api.add_middleware(BodyLimitMiddleware, limit_fn=lambda _path: 1 << 16)

    @api.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "device": separator.device,
            "pending": separator.pending,
            "max_pending": separator.config.max_pending,
        }

    @api.post("/separate")
    async def separate(body: SeparateRequest, request: Request) -> dict[str, Any]:
        started = time.time()
        if not (separator.config.contains(body.in_wav) and separator.config.contains(body.out_dir)):
            raise HTTPException(400, "path outside allowed root")
        separator.try_enter()
        abandoned = threading.Event()
        watcher = asyncio.create_task(_watch_disconnect(request, abandoned))
        loop = asyncio.get_running_loop()
        try:
            future = loop.run_in_executor(None, separator.work, body, abandoned, started)
        except BaseException:
            watcher.cancel()
            separator.leave()
            raise

        def finished(f: asyncio.Future[dict[str, Any]]) -> None:
            # The pending slot is freed when the THREAD ends, not when the await is cancelled.
            if not f.cancelled():
                f.exception()
            loop.call_soon_threadsafe(watcher.cancel)
            separator.leave()

        future.add_done_callback(finished)
        try:
            return await asyncio.shield(future)
        except HTTPException:
            raise
        except Abandoned:
            raise HTTPException(503, "client disconnected while queued", headers={"Retry-After": "5"}) from None
        except asyncio.CancelledError:
            abandoned.set()  # nobody is waiting any more: stop at the next checkpoint
            raise
        except Exception as exc:
            log.exception("separate failed")
            raise HTTPException(500, str(exc)) from exc

    return api


app = create_app(Separator(SeparatorConfig.from_env()))
