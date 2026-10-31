"""llm_client: the one place the dubbing app talks to an LLM.

LLM_PROVIDER=openrouter (default) | gemini.
OpenRouter: OpenAI-compatible POST https://openrouter.ai/api/v1/chat/completions, Bearer auth.
Key: env OPENROUTER_API_KEY, else the file named by env OPENROUTER_API_KEY_FILE. The key is never logged and never put in an
exception text (all errors carry only the HTTP status / exception class).

Models per use-case ("chat", "fit", "translate"), env override first:
  LLM_MODEL_<USECASE>  (any id, used verbatim)   e.g. LLM_MODEL_FIT=google/gemini-2.5-flash
  else DUB_CHAT_MODEL / GEMINI_MODEL / gemini-3.5-flash-lite,
  mapped to the OpenRouter id "google/<name>" when provider=openrouter (a name that already has '/' is kept).

Public API:
  generate_text(prompt, use_case="fit", ...)            -> str      (fit rewriter, en translation)
  chat_tools(system, contents, declarations, ...)       -> dict     (Gemini-shaped response, so ctl_chat.parse_response is unchanged)
  provider(), resolve_model(use_case), legacy_gemini_key()
Budget logging: one logger line per call (use_case, model, tokens, usd if the provider reports it) and, if LLM_USAGE_LOG is set,
one JSON line appended to that file. No prompt text and no key is ever written.
"""
import json
import logging
import os
import threading
import time

import requests

log = logging.getLogger("llm_client")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
_LOCK = threading.Lock()


class LLMError(RuntimeError):
    """Never contains a key, a URL with a query, or a response body: status / class only."""

    def __init__(self, msg, status=None):
        super().__init__(msg)
        self.status = status


def provider():
    """Return the configured LLM provider, ``openrouter`` (default) or ``gemini``."""
    p = (os.environ.get("LLM_PROVIDER") or "openrouter").strip().lower()
    if p not in ("gemini", "openrouter"):
        raise LLMError(f"LLM_PROVIDER must be openrouter|gemini, got {p!r}")
    return p


def _default_model(use_case):
    if use_case == "chat":
        return os.environ.get("DUB_CHAT_MODEL") or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    return os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL


def resolve_model(use_case, prov=None):
    """Return the model id for ``use_case`` (``chat``, ``fit`` or ``translate``) under the active provider."""
    prov = prov or provider()
    ov = (os.environ.get("LLM_MODEL_" + use_case.upper()) or "").strip()
    if ov:
        return ov
    m = _default_model(use_case)
    if prov == "openrouter" and "/" not in m:
        return "google/" + m
    return m


def _read_file(path):
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def openrouter_key():
    """Return the OpenRouter key from the environment or its key file; raises :class:`LLMError` when absent."""
    k = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if k:
        return k
    p = (os.environ.get("OPENROUTER_API_KEY_FILE") or os.environ.get("OPENROUTER_KEY_FILE") or "").strip()
    if p:
        try:
            k = _read_file(p)
        except (OSError, ValueError) as e:
            raise LLMError(f"cannot read OPENROUTER_API_KEY_FILE ({type(e).__name__})") from None
        if k:
            return k
    raise LLMError("no OpenRouter key: set OPENROUTER_API_KEY or OPENROUTER_API_KEY_FILE")


def gemini_key():
    """Return the Gemini key from the environment or its key file; raises :class:`LLMError` when absent."""
    k = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if k:
        return k
    try:
        return _read_file(os.environ.get("GEMINI_KEY_FILE") or os.environ.get("DUB_GEMINI_KEY_PATH") or "")
    except (OSError, ValueError) as e:
        raise LLMError(f"cannot read Gemini key file ({type(e).__name__})") from None


def legacy_gemini_key():
    """What old call sites pass around as `key`. Under openrouter nothing needs it -> ''. Never raises."""
    if provider() == "openrouter":
        return ""
    try:
        return gemini_key()
    except LLMError:
        return ""


def _headers(prov, key):
    if prov == "openrouter":
        h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        ref = os.environ.get("OPENROUTER_REFERER")
        if ref:
            h["HTTP-Referer"] = ref
        h["X-Title"] = os.environ.get("OPENROUTER_TITLE", "SunwAI dubbing")
        return h
    return {"x-goog-api-key": key}


def _record(use_case, model, prov, tin, tout, usd, status="ok"):
    rec = {"ts": round(time.time(), 3), "use_case": use_case, "provider": prov, "model": model,
           "in": tin, "out": tout, "usd": usd, "status": status}
    log.info("llm %s %s %s in=%s out=%s usd=%s %s", use_case, prov, model, tin, tout, usd, status)
    path = os.environ.get("LLM_USAGE_LOG")
    if path:
        try:
            with _LOCK, open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass


def _post(prov, model, body, key, timeout, session=None):
    url = OPENROUTER_URL if prov == "openrouter" else GEMINI_URL.format(model=model)
    try:
        return (session or requests).post(url, headers=_headers(prov, key), json=body, timeout=timeout)
    except requests.exceptions.RequestException as e:
        raise LLMError(f"LLM request failed ({type(e).__name__})") from None


def _openrouter_body(model, messages, temperature, max_tokens, response_format=None, tools=None):
    b = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens:
        b["max_tokens"] = int(max_tokens)
    if response_format:
        b["response_format"] = response_format
    if tools:
        b["tools"] = tools
        b["tool_choice"] = "auto"
    return b


def _rf(response_format):
    """response_format: None | 'json' | JSON-schema dict -> OpenRouter response_format object."""
    if not response_format:
        return None
    if response_format == "json":
        return {"type": "json_object"}
    return {"type": "json_schema", "json_schema": {"name": "out", "strict": True, "schema": response_format}}


def _retryable(status):
    return status == 429 or status >= 500


def generate_text(prompt, use_case="fit", *, temperature=0, timeout=60, retries=5, response_format=None, max_tokens=None,
                  key=None, sleep=time.sleep, session=None):
    """One plain-text completion (same retry shape as the old fit_stage._gemini: `retries` attempts, 2*2**a s backoff)."""
    prov = provider()
    model = resolve_model(use_case, prov)
    key = openrouter_key() if prov == "openrouter" else ((key or "").strip() or gemini_key())     # explicit key honoured for gemini only
    if prov == "openrouter":
        body = _openrouter_body(model, [{"role": "user", "content": prompt}], temperature, max_tokens, _rf(response_format))
    else:
        gc = {"temperature": temperature}
        if max_tokens:
            gc["maxOutputTokens"] = int(max_tokens)
        if response_format:
            gc["responseMimeType"] = "application/json"
            if isinstance(response_format, dict):
                gc["responseSchema"] = response_format
        body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gc}
    last = None
    for attempt in range(retries):
        r = _post(prov, model, body, key, timeout, session)
        if r.ok:
            try:
                j = r.json()
                if prov == "openrouter":
                    text = j["choices"][0]["message"]["content"] or ""
                    if isinstance(text, list):
                        text = "".join(x.get("text", "") for x in text if isinstance(x, dict))
                    if not isinstance(text, str):
                        raise TypeError("non-text content")
                    u = j.get("usage") or {}
                    _record(use_case, model, prov, u.get("prompt_tokens"), u.get("completion_tokens"), u.get("cost"))
                else:
                    text = j["candidates"][0]["content"]["parts"][0]["text"]
                    u = j.get("usageMetadata") or {}
                    _record(use_case, model, prov, u.get("promptTokenCount"), u.get("candidatesTokenCount"), None)
                return text.strip().strip('"')
            except (KeyError, IndexError, TypeError, ValueError, AttributeError):
                last = "malformed"
        else:
            last = r.status_code
            _record(use_case, model, prov, None, None, None, f"http_{last}")
            if r.status_code in (400, 401, 402, 403, 404):
                break
        if attempt < retries - 1:
            sleep(2 * 2 ** attempt)
    raise LLMError(f"LLM call failed (provider={prov}, last={last})", status=last if isinstance(last, int) else None)


def _to_openrouter_messages(system, contents):
    msgs = [{"role": "system", "content": system}] if system else []
    for c in contents:
        role = "assistant" if c.get("role") in ("model", "assistant") else "user"
        if any(set(p) - {"text"} for p in c.get("parts", [])):
            raise LLMError("openrouter adapter supports text-only turns (single-turn planner); got a non-text part")
        msgs.append({"role": role, "content": "".join(p.get("text", "") for p in c.get("parts", []))})
    return msgs


def _openrouter_to_gemini_shape(j):
    msg = ((j.get("choices") or [{}])[0].get("message")) or {}
    parts = []
    c = msg.get("content")
    if isinstance(c, list):
        c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
    if c and isinstance(c, str):
        parts.append({"text": c})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        raw = fn.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw if isinstance(raw, dict) else {})
        except ValueError:
            raise LLMError("malformed tool-call arguments")
        if not isinstance(fn.get("name"), str) or not isinstance(args, dict):
            raise LLMError("malformed tool call")
        parts.append({"functionCall": {"name": fn["name"], "args": args}})
    u = j.get("usage") or {}
    return {"candidates": [{"content": {"parts": parts}}],
            "usageMetadata": {"promptTokenCount": u.get("prompt_tokens", 0), "candidatesTokenCount": u.get("completion_tokens", 0)},
            "_cost": u.get("cost")}


def chat_tools(system, contents, declarations=None, *, use_case="chat", temperature=0, max_tokens=600, timeout=45, retries=2,
               attempt_cb=None, sleep=time.sleep, session=None):
    """Chat turn with optional function declarations (Gemini declaration format). Returns a Gemini-shaped dict either way.
    attempt_cb(nchars) runs before EVERY attempt (rate-gate + budget charge, exactly as ctl_chat did)."""
    prov = provider()
    model = resolve_model(use_case, prov)
    key = openrouter_key() if prov == "openrouter" else gemini_key()
    if prov == "openrouter":
        tools = [{"type": "function", "function": d} for d in (declarations or [])]
        body = _openrouter_body(model, _to_openrouter_messages(system, contents), temperature, max_tokens, None, tools)
    else:
        body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": contents,
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}}
        if declarations:
            body["tools"] = [{"functionDeclarations": declarations}]
            body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
    last = None
    for attempt in range(retries):
        if attempt_cb:
            attempt_cb(len(system) + sum(len(p.get("text", "")) for c in contents for p in c.get("parts", [])))
        r = _post(prov, model, body, key, timeout, session)
        if r.ok:
            try:
                j = r.json()
                if prov == "openrouter":
                    j = _openrouter_to_gemini_shape(j)
                um = j.get("usageMetadata") or {}
                _record(use_case, model, prov, um.get("promptTokenCount"), um.get("candidatesTokenCount"), j.get("_cost"))
                return j
            except (ValueError, TypeError, AttributeError, LLMError):
                last = "malformed"
        else:
            last = r.status_code
            _record(use_case, model, prov, None, None, None, f"http_{last}")
            if r.status_code in (400, 401, 402, 403, 404):
                break
        if attempt < retries - 1:
            sleep(1.5 * 2 ** attempt)
    raise LLMError(f"planner HTTP {last}", status=last if isinstance(last, int) else None)


class ConfiguredLLM:
    """:class:`sunwai.clients.interfaces.LLMClient` backed by the provider chosen with ``LLM_PROVIDER``."""

    def __init__(self, timeout: float = 60.0, retries: int = 5) -> None:
        self._timeout = timeout
        self._retries = retries

    def complete(self, prompt: str, use_case: str = "fit") -> str:
        """Return the model's text answer for ``prompt`` (see :func:`generate_text`)."""
        return generate_text(prompt, use_case, timeout=self._timeout, retries=self._retries)
