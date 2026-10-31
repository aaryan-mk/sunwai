"""Offline tests for the LLM provider switch and the Bodhan api/local endpoint switch (no network)."""
import importlib
import json
import logging

import pytest

SECRET = "test-llm-key-SENTINEL-123456"
BSECRET = "bodhan-SECRETSENTINEL-654321"

ENVS = ["OPENROUTER_API_KEY_FILE", "BODHAN_API_KEY", "BODHAN_ASR_KEY", "BODHAN_MT_KEY", "BODHAN_TTS_KEY", "GEMINI_KEY_FILE", "DEP_HOST", "LLM_PROVIDER", "OPENROUTER_API_KEY", "OPENROUTER_KEY_FILE", "GEMINI_API_KEY", "DUB_GEMINI_KEY_PATH", "GEMINI_MODEL",
        "DUB_CHAT_MODEL", "LLM_MODEL_CHAT", "LLM_MODEL_FIT", "LLM_MODEL_TRANSLATE", "LLM_USAGE_LOG", "BODHAN_MODE", "BODHAN_API_BASE",
        "ASR_BASE", "MT_BASE", "MT_BASES", "TTS_BASE", "BODHAN_ASR_KEY_FILE", "BODHAN_MT_KEY_FILE", "BODHAN_TTS_KEY_FILE", "BODHAN_KEY_DIR"]


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for e in ENVS:
        monkeypatch.delenv(e, raising=False)


class Resp:
    def __init__(self, code=200, js=None, content=b""):
        self.status_code, self._js, self.content, self.ok = code, js, content, code < 400

    def json(self):
        return self._js

    def raise_for_status(self):
        if not self.ok:
            import requests
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


class Sess:
    def __init__(self, *resps):
        self.calls, self.resps = [], list(resps)

    def post(self, url, **kw):
        self.calls.append((url, kw))
        r = self.resps.pop(0) if len(self.resps) > 1 else self.resps[0]
        if isinstance(r, Exception):
            raise r
        return r


def OR_OK(text="hi", cost=0.0001, tool=None):
    m = {"role": "assistant", "content": text}
    if tool:
        m["tool_calls"] = [{"id": "c1", "type": "function", "function": {"name": tool[0], "arguments": json.dumps(tool[1])}}]
    return Resp(200, {"choices": [{"message": m}], "usage": {"prompt_tokens": 11, "completion_tokens": 5, "cost": cost}})


GEM_OK = Resp(200, {"candidates": [{"content": {"parts": [{"text": '"hello"'}]}}], "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2}})


def llm():
    import sunwai.clients.llm as llm_client
    return importlib.reload(llm_client)


# ------------------------------------------------------------------ LLM
def test_default_provider_is_openrouter_and_gemini_request(monkeypatch):
    L = llm(); assert L.provider() == "openrouter"
    monkeypatch.setenv("LLM_PROVIDER", "gemini"); monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    assert L.provider() == "gemini"
    s = Sess(GEM_OK)
    assert L.generate_text("p", "fit", session=s) == "hello"
    url, kw = s.calls[0]
    assert url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent"
    assert kw["headers"] == {"x-goog-api-key": SECRET}
    assert kw["json"] == {"contents": [{"parts": [{"text": "p"}]}], "generationConfig": {"temperature": 0}}


def test_openrouter_request(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter"); monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    L = llm(); s = Sess(OR_OK("ok"))
    assert L.generate_text("p", "fit", session=s) == "ok"
    url, kw = s.calls[0]
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert kw["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert kw["json"]["model"] == "google/gemini-3.5-flash-lite"
    assert kw["json"]["messages"] == [{"role": "user", "content": "p"}] and kw["json"]["temperature"] == 0
    assert "response_format" not in kw["json"] and kw["timeout"] == 60


def test_model_per_usecase_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter"); monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("LLM_MODEL_CHAT", "google/gemini-2.5-pro"); L = llm()
    assert L.resolve_model("fit") == "google/gemini-2.5-flash"
    assert L.resolve_model("translate") == "google/gemini-2.5-flash"
    assert L.resolve_model("chat") == "google/gemini-2.5-pro"
    monkeypatch.setenv("LLM_PROVIDER", "gemini"); assert L.resolve_model("fit") == "gemini-2.5-flash"


def test_json_outputs(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET); monkeypatch.setenv("LLM_PROVIDER", "openrouter"); L = llm()
    s = Sess(OR_OK("{}")); L.generate_text("p", response_format="json", session=s)
    assert s.calls[0][1]["json"]["response_format"] == {"type": "json_object"}
    sch = {"type": "object", "properties": {"a": {"type": "string"}}}
    s = Sess(OR_OK("{}")); L.generate_text("p", response_format=sch, session=s)
    assert s.calls[0][1]["json"]["response_format"]["json_schema"]["schema"] == sch
    monkeypatch.setenv("LLM_PROVIDER", "gemini"); monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    s = Sess(GEM_OK); L.generate_text("p", response_format="json", session=s)
    assert s.calls[0][1]["json"]["generationConfig"]["responseMimeType"] == "application/json"


def test_key_from_file_and_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter"); L = llm()
    with pytest.raises(L.LLMError) as e:
        L.generate_text("p", session=Sess(OR_OK()))
    assert "OPENROUTER" in str(e.value)
    kf = tmp_path / "k"; kf.write_text(SECRET + "\n"); monkeypatch.setenv("OPENROUTER_KEY_FILE", str(kf))
    s = Sess(OR_OK()); L.generate_text("p", session=s)
    assert s.calls[0][1]["headers"]["Authorization"] == f"Bearer {SECRET}"
    monkeypatch.setenv("OPENROUTER_KEY_FILE", str(tmp_path / "nope"))
    with pytest.raises(L.LLMError) as e:
        L.generate_text("p", session=Sess(OR_OK()))
    assert SECRET not in str(e.value) and e.value.__suppress_context__


def test_retries_and_no_key_leak(monkeypatch, caplog, tmp_path):
    import requests
    monkeypatch.setenv("LLM_PROVIDER", "openrouter"); monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    ul = tmp_path / "u.jsonl"; monkeypatch.setenv("LLM_USAGE_LOG", str(ul)); L = llm()
    caplog.set_level(logging.DEBUG)
    s = Sess(Resp(429, {"error": SECRET}), Resp(500, {}), OR_OK("fine")); slept = []
    assert L.generate_text("p", session=s, sleep=slept.append) == "fine" and len(s.calls) == 3 and slept == [2, 4]
    s = Sess(Resp(401, {"error": {"message": "bad key " + SECRET}}))
    with pytest.raises(L.LLMError) as e:
        L.generate_text("p", session=s, sleep=slept.append)
    assert len(s.calls) == 1 and e.value.status == 401 and SECRET not in str(e.value) and SECRET not in repr(e.value)
    s = Sess(requests.exceptions.ConnectionError("boom " + SECRET))
    with pytest.raises(L.LLMError) as e:
        L.generate_text("p", session=s, sleep=slept.append, retries=1)
    assert SECRET not in str(e.value) and e.value.__suppress_context__ and e.value.__cause__ is None
    assert SECRET not in caplog.text and SECRET not in ul.read_text()
    rec = [json.loads(l) for l in ul.read_text().splitlines()]
    assert rec[0]["status"] == "http_429" and any(r["status"] == "ok" and r["usd"] == 0.0001 and r["in"] == 11 for r in rec)


def test_chat_tools_both_providers(monkeypatch):
    from sunwai.control import tools as ctl_tools
    importlib.invalidate_caches()
    from sunwai.control import chat as ctl_chat; importlib.reload(ctl_chat)
    decl = ctl_tools.declarations()[:2]
    contents = [{"role": "user", "parts": [{"text": "hello"}]}]
    # openrouter
    monkeypatch.setenv("LLM_PROVIDER", "openrouter"); monkeypatch.setenv("OPENROUTER_API_KEY", SECRET); L = llm()
    s = Sess(OR_OK("txt", tool=(decl[0]["name"], {"x": 1}))); seen = []
    j = L.chat_tools("SYS", contents, decl, attempt_cb=seen.append, session=s)
    url, kw = s.calls[0]; b = kw["json"]
    assert url.startswith("https://openrouter.ai/api/v1/chat") and kw["timeout"] == 45
    assert b["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hello"}]
    assert b["tools"][0] == {"type": "function", "function": decl[0]} and b["tool_choice"] == "auto" and b["max_tokens"] == 600
    text, call, n, usage = ctl_chat.parse_response(j)
    assert (text, call, n) == ("txt", {"name": decl[0]["name"], "args": {"x": 1}}, 1) and usage["in"] == 11 and usage["out"] == 5
    assert seen == [len("SYS") + 5]
    # gemini
    monkeypatch.setenv("LLM_PROVIDER", "gemini"); monkeypatch.setenv("GEMINI_API_KEY", SECRET); L = llm()
    s = Sess(GEM_OK); L.chat_tools("SYS", contents, decl, session=s)
    url, kw = s.calls[0]
    assert url.endswith("gemini-3.5-flash-lite:generateContent") and kw["headers"] == {"x-goog-api-key": SECRET}
    assert kw["json"]["systemInstruction"] == {"parts": [{"text": "SYS"}]} and kw["json"]["toolConfig"] == {"functionCallingConfig": {"mode": "AUTO"}}
    assert kw["json"]["tools"] == [{"functionDeclarations": decl}]


def test_ctl_chat_error_maps_to_ctlerror(monkeypatch):
    from sunwai.control import chat as ctl_chat; importlib.reload(ctl_chat)
    from sunwai.control.store import CtlError
    monkeypatch.setenv("LLM_PROVIDER", "openrouter"); monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp(403, {}))
    with pytest.raises(CtlError) as e:
        ctl_chat._call_gemini("S", [{"role": "user", "parts": [{"text": "x"}]}], False, set())
    assert e.value.code == "planner_unavailable" and "403" in str(e.value.detail) and SECRET not in str(e.value.detail)


def test_fit_stage_routes_through_client(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter"); monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    import requests
    calls = []
    monkeypatch.setattr(requests, "post", lambda url, **kw: (calls.append((url, kw)), OR_OK("short"))[1])
    import sunwai.pipeline.fitting.fit_stage as fit_stage; importlib.reload(fit_stage)
    assert fit_stage._llm_call("prompt", "ignored-key") == "short"
    assert calls[0][0] == "https://openrouter.ai/api/v1/chat/completions" and "ignored-key" not in json.dumps(calls[0][1]["headers"])
    assert importlib.import_module("sunwai.clients.llm").legacy_gemini_key() == ""      # no .gemini_key read under openrouter


# ------------------------------------------------------------------ Bodhan
def cfg():
    import sunwai.clients.bodhan_endpoints as bodhan_cfg
    return importlib.reload(bodhan_cfg)


def test_bodhan_cfg_modes(monkeypatch):
    c = cfg()
    assert c.mode() == "api" and c.asr_base() == "https://api.bodhan.ai"
    monkeypatch.setenv("BODHAN_MODE", "local")
    assert c.asr_base() == "http://127.0.0.1:18931"
    assert c.mt_bases()[0] == "http://127.0.0.1:18933" and len(c.mt_bases()) == 5 and len(c.tts_bases()) == 10
    monkeypatch.setenv("BODHAN_MODE", "api")
    assert c.asr_base() == "https://api.bodhan.ai" and c.mt_bases() == ["https://api.bodhan.ai"] * 5 and c.tts_bases() == ["https://api.bodhan.ai"] * 10
    monkeypatch.setenv("BODHAN_API_TTS_FANOUT", "3"); assert len(c.tts_bases()) == 3; monkeypatch.delenv("BODHAN_API_TTS_FANOUT")
    with pytest.raises(c.BodhanConfigError):
        c.load_keys()                                   # api mode needs a key
    monkeypatch.setenv("BODHAN_API_KEY", BSECRET); monkeypatch.setenv("BODHAN_MT_KEY", "mt-specific")
    assert c.load_keys() == {"asr": BSECRET, "mt": "mt-specific", "tts": BSECRET}
    monkeypatch.setenv("BODHAN_MODE", "local"); monkeypatch.delenv("BODHAN_API_KEY"); monkeypatch.delenv("BODHAN_MT_KEY")
    assert c.load_keys() == {"asr": "", "mt": "", "tts": ""}      # local replicas need no key
    monkeypatch.setenv("BODHAN_MODE", "weird")
    with pytest.raises(ValueError):
        c.mode()
    monkeypatch.setenv("BODHAN_MODE", "local"); monkeypatch.setenv("MT_BASE", "http://h:1"); monkeypatch.setenv("TTS_BASE", " http://t:1 ,http://t:2")
    assert c.mt_bases() == ["http://h:1"] and c.tts_bases() == ["http://t:1", "http://t:2"]
    assert "SECRET" not in json.dumps(c.summary())


def dp(monkeypatch, tmp_path, mode):
    monkeypatch.setenv("DUB_SLOT_DIR", str(tmp_path)); monkeypatch.setenv("BODHAN_MODE", mode)
    cfg()
    import sunwai.pipeline.engine as dub_pipeline
    return importlib.reload(dub_pipeline)


@pytest.mark.parametrize("mode,base_re", [("api", "https://api.bodhan.ai"), ("local", "http://127.0.0.1:")])
def test_bodhan_requests_both_modes(monkeypatch, tmp_path, mode, base_re):
    D = dp(monkeypatch, tmp_path, mode)
    wav = tmp_path / "a.wav"; wav.write_bytes(b"RIFFxxxx")
    hdr = {"Authorization": f"Bearer {BSECRET}"}
    s = Sess(Resp(200, {"text": " namaste "}))
    assert D.bodhan_asr(s, BSECRET, wav, "hi") == "namaste"
    u, kw = s.calls[0]
    assert u.startswith(base_re) and u.endswith("/v1/audio/transcriptions") and kw["headers"] == hdr and kw["data"] == {"model": "indic-transcribe", "language": "hi"}
    s = Sess(Resp(200, {"choices": [{"message": {"content": " X "}}]}))
    assert D.bodhan_mt(s, BSECRET, "hello", "hi") == "X"
    u, kw = s.calls[0]
    assert u.startswith(base_re) and u.endswith("/v1/chat/completions") and kw["headers"] == {**hdr, "Content-Type": "application/json"}
    assert kw["json"] == {"model": "indic-translate", "messages": [{"role": "user", "content": "hello"}], "target_language": "hi"}
    s = Sess(Resp(200, content=b"WAVDATA"))
    assert D.bodhan_tts(s, BSECRET, "hi", "hi", "Amit", style="calm") == b"WAVDATA"
    u, kw = s.calls[0]
    assert u.startswith(base_re) and u.endswith("/v1/audio/speech") and kw["headers"] == {**hdr, "Content-Type": "application/json"}
    assert kw["json"] == {"model": "indic-speak", "input": "hi", "voice": "Amit", "instructions": json.dumps({"lang": "hi", "style": "calm"})}
    if mode == "api":
        assert u == "https://api.bodhan.ai/v1/audio/speech"


def test_bodhan_error_has_no_key(monkeypatch, tmp_path, caplog):
    import requests
    D = dp(monkeypatch, tmp_path, "api"); caplog.set_level(logging.DEBUG)
    s = Sess(Resp(401, {}))
    with pytest.raises(Exception) as e:
        D.bodhan_mt(s, BSECRET, "t", "hi")
    s = Sess(requests.exceptions.ConnectionError("down"))
    monkeypatch.setenv("DUB_RETRIES", "0")
    with pytest.raises(Exception) as e2:
        D.bodhan_mt(s, BSECRET, "t", "hi", retries=0)
    assert BSECRET not in str(e.value) + str(e2.value) + caplog.text


def test_routed_mt_en_uses_llm_client(monkeypatch, tmp_path):
    D = dp(monkeypatch, tmp_path, "local")
    monkeypatch.setenv("LLM_PROVIDER", "openrouter"); monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    import requests
    calls = []
    monkeypatch.setattr(requests, "post", lambda url, **kw: (calls.append((url, kw)), OR_OK("hello"))[1])
    assert D.routed_mt(None, "k", "namaste", "en", "hi") == "hello"
    assert calls[0][0].startswith("https://openrouter.ai/") and calls[0][1]["json"]["model"] == "google/gemini-3.5-flash-lite"


def test_adapter_rejects_non_text_and_bad_tool_args(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter"); monkeypatch.setenv("OPENROUTER_API_KEY", SECRET); L = llm()
    with pytest.raises(L.LLMError):
        L.chat_tools("S", [{"role": "user", "parts": [{"functionResponse": {}}]}], [], session=Sess(OR_OK()))
    bad = Resp(200, {"choices": [{"message": {"content": None, "tool_calls": [{"function": {"name": "x", "arguments": "{not json"}}]}}]})
    with pytest.raises(L.LLMError):
        L.chat_tools("S", [{"role": "user", "parts": [{"text": "x"}]}], [], session=Sess(bad))
    lst = Resp(200, {"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}]})
    assert L.chat_tools("S", [{"role": "user", "parts": [{"text": "x"}]}], [], session=Sess(lst))["candidates"][0]["content"]["parts"] == [{"text": "ab"}]


def test_explicit_gemini_key_honoured(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    L = llm(); s = Sess(GEM_OK)
    L.generate_text("p", key="explicit-k", session=s)
    assert s.calls[0][1]["headers"] == {"x-goog-api-key": "explicit-k"}
