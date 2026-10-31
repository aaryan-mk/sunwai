"""IndicTrans2 MT server with the same HTTP contract as ``mt_server.py`` (indic-translate mirror).

``POST /v1/chat/completions {"messages": [{"role": "user", "content": TEXT}], "target_language": "hi",
"source_language": "en"}``. English to Indic uses ``indictrans2-en-indic-1B``; Indic to Indic uses
``indictrans2-indic-indic-1B``. Pre/post-processing via IndicTransToolkit; beam 5. Requests are serialised and
bounded by an admission limit.

Run with ``python -m sunwai.mirrors.indictrans2 --models <dir> --port 18936``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
import uuid
from collections.abc import Callable
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from IndicTransToolkit.processor import IndicProcessor
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from sunwai.mirrors.admission import Admission, Limits
from sunwai.mirrors.body_limit import BodyLimitMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("it2_server")

FLORES = {
    "en": "eng_Latn", "hi": "hin_Deva", "ta": "tam_Taml", "bn": "ben_Beng", "mr": "mar_Deva", "te": "tel_Telu",
    "gu": "guj_Gujr", "kn": "kan_Knda", "ml": "mal_Mlym", "ur": "urd_Arab", "pa": "pan_Guru", "or": "ory_Orya",
    "as": "asm_Beng", "brx": "brx_Deva", "doi": "doi_Deva", "ks": "kas_Arab", "kok": "gom_Deva",
    "mai": "mai_Deva", "mni": "mni_Mtei", "ne": "npi_Deva", "sa": "san_Deva", "sat": "sat_Olck", "sd": "snd_Deva",
}  # fmt: skip
DEFAULT_MAX_TOKENS = 256


class Translator:
    """The two IndicTrans2 models plus the shared pre/post-processor."""

    def __init__(self) -> None:
        """Create an unloaded translator."""
        self.device: str | None = None
        self.processor: Any = None
        self.models: dict[str, tuple[Any, Any]] = {}

    def load(self, models_dir: str, device: str) -> None:
        """Load both checkpoints from ``models_dir`` onto ``device``."""
        self.device = device
        self.processor = IndicProcessor(inference=True)
        self.models["en-indic"] = self._load_model(f"{models_dir}/indictrans2-en-indic-1B", device)
        self.models["indic-indic"] = self._load_model(f"{models_dir}/indictrans2-indic-indic-1B", device)

    @staticmethod
    def _load_model(model_dir: str, device: str) -> tuple[Any, Any]:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=dtype)
        return tokenizer, model.to(device).eval()

    def translate(self, texts: list[str], src: str, tgt: str, max_tokens: int | None) -> list[str]:
        """Translate ``texts`` from ``src`` to ``tgt`` (ISO codes) with beam search 5."""
        src_code, tgt_code = FLORES.get(src, src), FLORES.get(tgt, tgt)
        tokenizer, model = self.models["en-indic" if src_code == "eng_Latn" else "indic-indic"]
        batch = self.processor.preprocess_batch(texts, src_lang=src_code, tgt_lang=tgt_code)
        inputs = tokenizer(batch, padding="longest", truncation=True, max_length=256, return_tensors="pt").to(
            self.device
        )
        with torch.inference_mode():
            generated = model.generate(
                **inputs, num_beams=5, max_length=max_tokens or DEFAULT_MAX_TOKENS, use_cache=True
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        result: list[str] = self.processor.postprocess_batch(decoded, lang=tgt_code)
        return result


async def run_exclusive(
    request: Request, lock: asyncio.Lock, fn: Callable[..., list[str]], *args: Any, on_done: Callable[[], None]
) -> list[str]:
    """Run ``fn`` in the executor under ``lock``.

    The lock and the pending slot (via ``on_done``) are released only when the THREAD finishes, even if this request
    is cancelled or times out meanwhile: a cancelled await must never free capacity that a still-running inference
    occupies. A client that left while queued is skipped without starting the thread.
    """
    loop = asyncio.get_running_loop()
    held = started = False
    try:
        await lock.acquire()
        held = True
        if await request.is_disconnected():
            raise HTTPException(503, "client disconnected while queued", headers={"Retry-After": "2"})
        future = loop.run_in_executor(None, fn, *args)
        started = True

        def finished(f: asyncio.Future[list[str]]) -> None:
            if not f.cancelled():
                f.exception()  # mark retrieved
            lock.release()
            on_done()

        future.add_done_callback(finished)
    finally:
        if not started:
            if held:
                lock.release()
            on_done()
    return await asyncio.shield(future)


class Msg(BaseModel):
    """One chat message."""

    role: str
    content: str


class ChatReq(BaseModel):
    """OpenAI-style chat request with the translation languages."""

    model: str = "indictrans2"
    messages: list[Msg]
    target_language: str
    source_language: str | None = "en"
    script: str | None = None
    max_tokens: int | None = DEFAULT_MAX_TOKENS


class BatchReq(BaseModel):
    """Batch translation request."""

    texts: list[str]
    target_language: str
    source_language: str | None = "en"


def create_app(translator: Translator, limits: Limits) -> FastAPI:
    """Build the FastAPI application serving ``translator`` under ``limits``."""
    api = FastAPI()
    api.add_middleware(BodyLimitMiddleware, limit_fn=lambda _path: limits.max_body)
    admission = Admission(limits)
    lock = asyncio.Lock()

    @api.post("/v1/chat/completions")
    async def chat(req: ChatReq, request: Request) -> dict[str, Any]:
        text = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        if not text:
            raise HTTPException(400, "no user message")
        admission.check([text])
        max_tokens = max(1, min(int(req.max_tokens or DEFAULT_MAX_TOKENS), limits.max_new_tokens))
        src = req.source_language or "en"
        if req.target_language == "en" or FLORES.get(req.target_language) is None:
            raise HTTPException(400, f"unsupported target {req.target_language}")
        started = time.time()
        admission.pending += 1
        out = await run_exclusive(
            request, lock, translator.translate, [text], src, req.target_language, max_tokens, on_done=admission.release
        )
        log.info("it2 %s->%s %d chars -> %.2fs", src, req.target_language, len(text), time.time() - started)
        return {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "model": "indictrans2",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": out[0]}, "finish_reason": "stop"}],
        }

    @api.post("/v1/translate_batch")
    async def batch(req: BatchReq, request: Request) -> dict[str, Any]:
        if not req.texts or len(req.texts) > 32:
            raise HTTPException(413, "texts must hold 1..32 items")
        admission.check(req.texts)
        admission.pending += 1
        out = await run_exclusive(
            request,
            lock,
            translator.translate,
            req.texts,
            req.source_language or "en",
            req.target_language,
            DEFAULT_MAX_TOKENS,
            on_done=admission.release,
        )
        return {"translations": out}

    @api.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "mt-indictrans2",
            "device": translator.device,
            "pending": admission.pending,
            "max_pending": limits.max_pending,
        }

    return api


TRANSLATOR = Translator()
app = create_app(TRANSLATOR, Limits.from_env())


def main() -> None:
    """Load the models and serve on ``0.0.0.0``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=os.environ.get("IT2_MODELS_DIR", "models"))
    parser.add_argument("--port", type=int, default=18936)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    TRANSLATOR.load(args.models, args.device)
    log.info("serving indictrans2 on 0.0.0.0:%d", args.port)
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
