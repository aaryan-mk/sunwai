"""Planner model access: the LLM call, price estimates and response parsing."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from sunwai.control import tools as ctl_tools
from sunwai.control.errors import CtlError

MODEL = os.environ.get("DUB_CHAT_MODEL", os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"))
# ASSUMED list prices ($/MTok) for the cost column; override by env. Not verified against the bill.
PRICE_IN = float(os.environ.get("DUB_CHAT_PRICE_IN", "0.10"))
PRICE_OUT = float(os.environ.get("DUB_CHAT_PRICE_OUT", "0.40"))
MAX_OUTPUT_TOKENS = 600


def estimate_cost_usd(prompt_chars: int) -> float:
    """Estimate the cost of one planner call from the prompt size (about 3 characters per token) and the output cap."""
    return (prompt_chars / 3.0) * PRICE_IN / 1e6 + MAX_OUTPUT_TOKENS * PRICE_OUT / 1e6


def call_planner(
    system: str,
    user_parts: list[dict[str, Any]],
    tools_on: bool,
    available: set[str],
    attempt_cb: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Call the planner through the LLM client (``LLM_PROVIDER=gemini|openrouter``); the answer is Gemini-shaped either way.

    Raises:
        CtlError: 503 ``planner_unavailable`` when the LLM client fails.
    """
    from sunwai.clients import llm as llm_client

    declarations = [d for d in ctl_tools.declarations() if d["name"] in available] if tools_on else []
    try:
        return llm_client.chat_tools(  # type: ignore[no-any-return]
            system,
            user_parts,
            declarations,
            use_case="chat",
            temperature=0,
            max_tokens=MAX_OUTPUT_TOKENS,
            timeout=45,
            retries=2,
            attempt_cb=attempt_cb,
        )
    except llm_client.LLMError as exc:
        raise CtlError(503, "planner_unavailable", f"planner HTTP {exc.status}") from exc


def parse_response(response: dict[str, Any]) -> tuple[str, dict[str, Any] | None, int, dict[str, Any]]:
    """Parse a Gemini-shaped response; only the FIRST function call is ever used.

    Returns:
        ``(text, first_call_or_None, number_of_calls, usage)`` with ``usage = {"in", "out", "usd"}``.
    """
    candidates = response.get("candidates") or []
    parts = ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
    text = "".join(p.get("text", "") for p in parts if "text" in p).strip()
    calls = [p["functionCall"] for p in parts if "functionCall" in p]
    meta = response.get("usageMetadata") or {}
    tokens_in = int(meta.get("promptTokenCount", 0))
    tokens_out = int(meta.get("candidatesTokenCount", 0)) + int(meta.get("thoughtsTokenCount", 0))
    usage = {
        "in": tokens_in,
        "out": tokens_out,
        "usd": round(tokens_in * PRICE_IN / 1e6 + tokens_out * PRICE_OUT / 1e6, 6),
    }
    call = {"name": calls[0].get("name"), "args": dict(calls[0].get("args") or {})} if calls else None
    return text, call, len(calls), usage
