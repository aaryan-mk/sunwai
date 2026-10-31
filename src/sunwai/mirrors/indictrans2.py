"""IndicTrans2 MT server with the same HTTP contract as mt_server.py (indic-translate mirror).

POST /v1/chat/completions {"messages":[{"role":"user","content":TEXT}],"target_language":"hi","source_language":"en"}
en -> Indic uses indictrans2-en-indic-1B; Indic -> Indic uses indictrans2-indic-indic-1B.
Pre/post-processing via IndicTransToolkit; beam 5. Requests are micro-batched per (src, tgt) pair.
"""
import argparse
import asyncio
import os
import logging
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from IndicTransToolkit.processor import IndicProcessor
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("it2_server")

FLORES = {"en": "eng_Latn", "hi": "hin_Deva", "ta": "tam_Taml", "bn": "ben_Beng", "mr": "mar_Deva", "te": "tel_Telu",
          "gu": "guj_Gujr", "kn": "kan_Knda", "ml": "mal_Mlym", "ur": "urd_Arab", "pa": "pan_Guru", "or": "ory_Orya",
          "as": "asm_Beng", "brx": "brx_Deva", "doi": "doi_Deva", "ks": "kas_Arab", "kok": "gom_Deva",
          "mai": "mai_Deva", "mni": "mni_Mtei", "ne": "npi_Deva", "sa": "san_Deva", "sat": "sat_Olck", "sd": "snd_Deva"}

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

app.add_middleware(_BodyLimit, limit_fn=lambda p: int(os.environ.get("IT2_MAX_BODY", str(2 << 20))))
STATE = {}
LOCK = asyncio.Lock()
MAX_PENDING = int(os.environ.get("IT2_MAX_PENDING", "32"))      # in-flight (running + queued) requests; more get 503
MAX_CHARS = int(os.environ.get("IT2_MAX_CHARS", "4000"))
MAX_NEW_TOKENS = int(os.environ.get("IT2_MAX_TOKENS", "1024"))
PENDING = [0]


async def _guarded(request, lock, fn, *args, on_done=None):
    """Run fn in the executor under `lock`. The lock (and the pending slot, via on_done) are released only when the
    THREAD finishes, even if this request is cancelled/times out meanwhile: a cancelled await must never free capacity
    that a still-running inference occupies. A client that left while queued is skipped without starting the thread."""
    loop = asyncio.get_running_loop()
    held = started = False
    try:
        await lock.acquire()
        held = True
        if await request.is_disconnected():
            raise HTTPException(503, "client disconnected while queued", headers={"Retry-After": "2"})
        fut = loop.run_in_executor(None, fn, *args)
        started = True

        def _done(f):
            if not f.cancelled():
                f.exception()                                # mark retrieved
            lock.release()
            if on_done:
                on_done()
        fut.add_done_callback(_done)
    finally:
        if not started:
            if held:
                lock.release()
            if on_done:
                on_done()
    return await asyncio.shield(fut)


def _dec():
    PENDING[0] -= 1


def _admit(texts):
    for t in texts:
        if len(t) > MAX_CHARS:
            raise HTTPException(413, f"text too long ({len(t)} chars; limit {MAX_CHARS})")
    if PENDING[0] >= MAX_PENDING:
        raise HTTPException(503, f"it2 service is overloaded ({MAX_PENDING} requests queued); retry shortly",
                            headers={"Retry-After": "2"})


class Msg(BaseModel):
    role: str
    content: str


class ChatReq(BaseModel):
    model: str = "indictrans2"
    messages: list[Msg]
    target_language: str
    source_language: str | None = "en"
    script: str | None = None
    max_tokens: int | None = 256


def load(model_dir, device):
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, trust_remote_code=True,
                                                  torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32).to(device).eval()
    return tok, model


def translate(texts, src, tgt, max_tokens):
    s, t = FLORES.get(src, src), FLORES.get(tgt, tgt)
    tok, model = STATE["en-indic"] if s == "eng_Latn" else STATE["indic-indic"]
    ip = STATE["ip"]
    batch = ip.preprocess_batch(texts, src_lang=s, tgt_lang=t)
    x = tok(batch, padding="longest", truncation=True, max_length=256, return_tensors="pt").to(STATE["device"])
    with torch.inference_mode():
        out = model.generate(**x, num_beams=5, max_length=max_tokens or 256, use_cache=True)
    out = tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return ip.postprocess_batch(out, lang=t)


@app.post("/v1/chat/completions")
async def chat(req: ChatReq, request: Request):
    text = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    if not text:
        raise HTTPException(400, "no user message")
    _admit([text])
    req.max_tokens = max(1, min(int(req.max_tokens or 256), MAX_NEW_TOKENS))
    src = req.source_language or "en"
    if req.target_language == "en" or FLORES.get(req.target_language) is None:
        raise HTTPException(400, f"unsupported target {req.target_language}")
    t0 = time.time()
    PENDING[0] += 1
    out = await _guarded(request, LOCK, translate, [text], src, req.target_language, req.max_tokens, on_done=_dec)
    log.info("it2 %s->%s %d chars -> %.2fs", src, req.target_language, len(text), time.time() - t0)
    return {"id": str(uuid.uuid4()), "object": "chat.completion", "model": "indictrans2",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": out[0]}, "finish_reason": "stop"}]}


class BatchReq(BaseModel):
    texts: list[str]
    target_language: str
    source_language: str | None = "en"


@app.post("/v1/translate_batch")
async def batch(req: BatchReq, request: Request):
    if not req.texts or len(req.texts) > 32:
        raise HTTPException(413, "texts must hold 1..32 items")
    _admit(req.texts)
    PENDING[0] += 1
    out = await _guarded(request, LOCK, translate, req.texts, req.source_language or "en", req.target_language, 256, on_done=_dec)
    return {"translations": out}


@app.get("/health")
def health():
    return {"ok": True, "service": "mt-indictrans2", "device": STATE.get("device"), "pending": PENDING[0], "max_pending": MAX_PENDING}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.environ.get("IT2_MODELS_DIR", "models"))
    ap.add_argument("--port", type=int, default=18936)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    STATE["device"] = a.device
    STATE["ip"] = IndicProcessor(inference=True)
    STATE["en-indic"] = load(f"{a.models}/indictrans2-en-indic-1B", a.device)
    STATE["indic-indic"] = load(f"{a.models}/indictrans2-indic-indic-1B", a.device)
    log.info("serving indictrans2 on 0.0.0.0:%d", a.port)
    uvicorn.run(app, host="0.0.0.0", port=a.port)
