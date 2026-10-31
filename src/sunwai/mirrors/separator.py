"""GPU htdemucs separation service (one model resident, requests serialized).
POST /separate {"in_wav": ".../src44k.wav", "out_dir": "..."} -> writes out_dir/vocals.wav, bed.wav (same math as dub_pipeline.stage_separate)."""
import os, asyncio, threading, time, logging
import numpy as np
import torch, librosa, soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import demucs.pretrained as dp
import demucs.apply as da

log = logging.getLogger("sep_server"); logging.basicConfig(level=logging.INFO)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = dp.get_model("htdemucs").to(DEV).eval()
LOCK = threading.Lock()
MAX_PENDING = int(os.environ.get("SEP_MAX_PENDING", "24"))
QUEUE_TIMEOUT = float(os.environ.get("SEP_QUEUE_TIMEOUT", "900"))
_PEND = [0]
MAX_BYTES = int(os.environ.get("SEP_MAX_BYTES", str(256 * 1024 * 1024)))   # input wav size ceiling (a 100 s stereo 44.1 kHz chunk is ~17 MB)
DEC = threading.Semaphore(int(os.environ.get("SEP_DECODE_PARALLEL", "2")))  # bounded concurrent decodes (memory)
_PLOCK = threading.Lock()
ROOT = os.path.realpath(os.environ.get("SEP_ROOT", os.getcwd()))


def _inside(p):
    rp = os.path.realpath(p)
    return rp == ROOT or rp.startswith(ROOT.rstrip(os.sep) + os.sep)    # component-wise: /amk-evil does not match /amk
app = FastAPI()


# ---- refuse absurd request bodies from the Content-Length header, before anything is parsed or buffered
class _BodyTooLarge(BaseException):          # BaseException: endpoint code with `except Exception` must not swallow it
    pass


class _BodyLimit:
    def __init__(self, app, limit_fn):
        self.app, self.limit_fn = app, limit_fn

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self.limit_fn(scope.get("path", ""))
        from starlette.responses import JSONResponse as _JR

        async def too_large(n):
            await _JR({"detail": f"request body too large ({n} bytes; limit {limit})"}, status_code=413)(scope, receive, send)

        for k, v in scope.get("headers", []):
            if k == b"content-length":
                try:
                    n = int(v)
                except ValueError:
                    n = 0
                if n > limit:
                    await too_large(n)
                    return
        seen = [0]
        started = [False]

        async def counting_receive():                 # also bounds chunked uploads (no Content-Length)
            msg = await receive()
            if msg.get("type") == "http.request":
                seen[0] += len(msg.get("body", b""))
                if seen[0] > limit:
                    raise _BodyTooLarge()
            return msg

        async def tracking_send(msg):
            if msg.get("type") == "http.response.start":
                started[0] = True
            await send(msg)
        try:
            await self.app(scope, counting_receive, tracking_send)
        except _BodyTooLarge:
            if not started[0]:
                await too_large(seen[0])

app.add_middleware(_BodyLimit, limit_fn=lambda p: 1 << 16)

class Req(BaseModel):
    in_wav: str
    out_dir: str

@app.get("/health")
def health():
    return {"ok": True, "device": DEV, "pending": _PEND[0], "max_pending": MAX_PENDING}

class _Abandoned(Exception):
    pass


def _check(ev):
    if ev.is_set():
        raise _Abandoned()


def _work(r, ev, t0):
    """Runs in a worker thread. ev is set by the async watcher when the client went away."""
    try:
        size = os.path.getsize(r.in_wav)
    except OSError:
        raise HTTPException(404, "input wav not found")
    if size > MAX_BYTES:
        raise HTTPException(413, f"input wav too large ({size} bytes; limit {MAX_BYTES})")
    _check(ev)
    with DEC:
        _check(ev)
        y, sr = librosa.load(r.in_wav, sr=44100, mono=False, dtype="float32")
    if y.ndim == 1:
        y = np.stack([y, y])
    sig = torch.from_numpy(y).unsqueeze(0)
    t_end = time.time() + QUEUE_TIMEOUT
    while not LOCK.acquire(timeout=0.25):                # poll so an abandoned/expired request leaves the queue promptly
        _check(ev)
        if time.time() > t_end:
            raise HTTPException(503, "separation queue wait exceeded the deadline", headers={"Retry-After": "5"})
    try:
        _check(ev)
        with torch.no_grad():
            out = da.apply_model(MODEL, sig, split=True, overlap=0.25, device=DEV)
            stems = out[0].cpu().numpy()
        names = list(MODEL.sources)
        v = names.index("vocals") if "vocals" in names else len(names) - 1
        voc, bed = stems[v], stems.sum(axis=0) - stems[v]
        od = r.out_dir
        os.makedirs(od, exist_ok=True)
        tmps = []
        uniq = f"{os.getpid()}.{threading.get_ident()}.{time.time_ns()}"
        try:
            for name, arr in (("vocals", voc), ("bed", bed)):
                tmp = f"{od}/.{name}.{uniq}.tmp.wav"
                tmps.append((tmp, f"{od}/{name}.wav"))
                sf.write(tmp, arr.T, 44100)
            for tmp, dst in tmps:      # publication stays INSIDE the lock: vocals+bed of two requests can never interleave
                os.replace(tmp, dst)
        finally:
            for tmp, _ in tmps:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    finally:
        LOCK.release()
    log.info("separated %s in %.1fs", r.in_wav, time.time() - t0)
    return {"ok": True, "secs": time.time() - t0}


@app.post("/separate")
async def separate(r: Req, request: Request):
    t0 = time.time()
    if not (_inside(r.in_wav) and _inside(r.out_dir)):
        raise HTTPException(400, "path outside allowed root")
    with _PLOCK:
        if _PEND[0] >= MAX_PENDING:
            raise HTTPException(503, f"separation service is overloaded ({MAX_PENDING} queued); retry shortly",
                                headers={"Retry-After": "5"})
        _PEND[0] += 1
    ev = threading.Event()

    async def watch():
        while True:
            await asyncio.sleep(0.25)
            try:
                if await request.is_disconnected():
                    ev.set()
                    return
            except Exception:
                return
    w = asyncio.create_task(watch())
    loop = asyncio.get_running_loop()
    try:
        fut = loop.run_in_executor(None, _work, r, ev, t0)
    except BaseException:
        w.cancel()
        with _PLOCK:
            _PEND[0] -= 1
        raise

    def _done(f):                                     # the pending slot is freed when the THREAD ends, not when the await is cancelled
        if not f.cancelled():
            f.exception()
        loop.call_soon_threadsafe(w.cancel)
        with _PLOCK:
            _PEND[0] -= 1
    fut.add_done_callback(_done)
    try:
        return await asyncio.shield(fut)
    except HTTPException:
        raise
    except _Abandoned:
        raise HTTPException(503, "client disconnected while queued", headers={"Retry-After": "5"})
    except asyncio.CancelledError:
        ev.set()                                      # nobody is waiting any more: stop at the next checkpoint
        raise
    except Exception as e:
        log.exception("separate failed")
        raise HTTPException(500, str(e))
