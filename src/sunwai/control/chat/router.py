"""Chat router: one bounded tool call per turn; the server validates, authorises and executes it."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from sunwai.control import hooks as ctl_hooks
from sunwai.control import tools as ctl_tools
from sunwai.control.chat.explain import explain_action
from sunwai.control.chat.planner import MODEL, call_planner, estimate_cost_usd, parse_response
from sunwai.control.chat.prompt import build_system_prompt, build_user_parts
from sunwai.control.chat.quotes import restore_verbatim, retime_out_of_scope
from sunwai.control.chat.snapshot import build_snapshot, resolve_segment
from sunwai.control.errors import CtlError
from sunwai.control.service import ActionService
from sunwai.control.settings import addlang_enabled, limits
from sunwai.control.store import JobStore, store_for
from sunwai.control.tools import CLIENT_TOOLS, BadArgs, validate_args

Event = tuple[str, dict[str, Any]]
TEXT_CHUNK = 24
PREVIEW_TOOLS = ("change_expression", "retime")
RETIME_REFUSAL = (
    "I can only change a line's speed within about 8 percent (0.92 to 1.08 times). "
    "That request is beyond what is supported, so I did not change anything."
)


def text_events(text: str) -> list[Event]:
    """Split ``text`` into streaming ``text`` events of at most :data:`TEXT_CHUNK` characters."""
    return [("text", {"delta": text[i : i + TEXT_CHUNK]}) for i in range(0, len(text), TEXT_CHUNK)]


def format_tempo(value: float) -> str:
    """Format a tempo factor with at most three decimals and no trailing zeros."""
    return ("%.3f" % value).rstrip("0").rstrip(".")


def error_events(exc: CtlError, tool: str) -> list[Event]:
    """Map a control-plane error raised while routing ``tool`` onto the friendly event the user should see."""
    if exc.status == 501:
        return [("text", {"delta": exc.detail if exc.code == "not_implemented" else "That is not available yet."})]
    if exc.code == "busy" and tool == "add_language":
        message = "The studio is busy with other dubbing jobs right now, so I could not start that language. Please try again in a few minutes."
        return [("text", {"delta": message})]
    if exc.code in ("busy", "one_synthesis_per_job", "one_synthesis_global"):
        return [("text", {"delta": "Another change is still being processed. Please try again in a moment."})]
    if exc.code == "schedule_failed":
        return [("text", {"delta": "I could not schedule that language just now (nothing was changed). Please try again shortly."})]
    return [("error", {"code": exc.code, "detail": exc.detail})]


def unavailable_text(tool: str) -> str:
    """Return the apology for a tool that is not currently offered."""
    if tool == "add_language":
        return "Adding a language while watching is not available yet."
    return f"Sorry, {tool.replace('_', ' ')} is not available yet. I can edit or re-express individual lines, switch language or subtitles."


def submission_args(name: str, args: dict[str, Any], context: dict[str, Any], lang: str | None, segment: tuple[int, str] | None) -> dict[str, Any]:
    """Build the ``args`` of the action submitted for tool ``name``."""
    if segment is not None:
        return {**args, "lang": lang, "chunk": segment[0], "segment_id": segment[1]}
    if name == "regenerate_chunk":
        return {"lang": lang, "chunk": args["chunk_id"]}
    if name == "add_language":
        return {**args, "playback_generation": int(context.get("playback_generation") or 0)}
    return dict(args)


class ChatRouter:
    """Plans a chat turn with the model, then routes its single tool call to the client or the action service."""

    def __init__(self, svc: ActionService, deps: Any) -> None:
        """Bind the action service and the host dependencies."""
        self.svc = svc
        self.d = deps

    def available(self) -> tuple[set[str], list[str]]:
        """Return the offered tool names and the labels of what is unavailable right now."""
        offered = set(ctl_tools.TOOLS)
        unavailable = ["whole-chunk regeneration"]
        offered.discard("regenerate_chunk")
        if not addlang_enabled():
            offered.discard("add_language")
            unavailable.append("adding a language while the video is playing (tell the user it is not available yet)")
        return offered, unavailable

    def _attempt_callback(self, store: JobStore, caps: dict[str, Any]) -> Callable[[int], None]:
        """Build the per-attempt callback that gates and charges every planner call, success or not."""
        guard = self.svc.guard

        def attempt(prompt_chars: int) -> None:
            estimate = estimate_cost_usd(prompt_chars)
            if guard:
                guard.gemini_gate()
                guard.check_daily(need_usd=estimate)
            store.check_caps(caps["spend_cap_usd"], caps["synth_cap_s"])
            store.add_spend(usd=estimate)
            if guard:
                guard.charge(usd=estimate, gemini_calls=1)

        return attempt

    async def plan(self, job_id: str, message: str, history: Any, context: dict[str, Any], tools_on: bool = True) -> dict[str, Any]:
        """Ask the planner what to do. Client-supplied ``history`` is ignored entirely (it would be attacker-controlled)."""
        job_dir = self.d.job_dir(job_id)
        langs = self.d.job_langs(job_dir)
        store = store_for(job_dir)
        caps = limits()
        store.check_caps(caps["spend_cap_usd"], caps["synth_cap_s"])
        snap = build_snapshot(lambda lang: self.d.all_segments(job_dir, lang), store, context, langs)
        offered, unavailable = self.available()
        system = build_system_prompt(unavailable, "add_language" in offered)
        parts = build_user_parts(message, context, snap)
        response = await asyncio.to_thread(call_planner, system, parts, tools_on, offered, self._attempt_callback(store, caps))
        text, call, ncalls, usage = parse_response(response)
        return {"text": text, "call": call, "ncalls": ncalls, "usage": usage, "snap": snap, "available": offered}

    async def turn(self, job_id: str, message: str, history: Any, context: dict[str, Any], tools_on: bool = True) -> AsyncIterator[Event]:
        """Run one chat turn as an async generator of ``(event, data)`` for the ``/chat`` SSE."""
        turn_id = "turn_" + uuid.uuid4().hex[:10]
        try:
            plan = await self.plan(job_id, message, history, context, tools_on=tools_on)
        except CtlError as exc:
            yield "error", {"code": exc.code, "detail": exc.detail}
            return
        except Exception as exc:
            yield "error", {"code": "planner_unavailable", "detail": type(exc).__name__}
            return
        for event in text_events(plan["text"]):
            yield event
        call = plan["call"]
        if not call and not plan["text"]:
            yield "text", {"delta": "I did not take any action. Could you rephrase what you would like to change?"}
        if call:
            try:
                async for event in self.route(job_id, turn_id, call, context, plan["snap"], plan["available"], message):
                    yield event
            except Exception as exc:
                yield "error", {"code": "internal", "detail": type(exc).__name__}
        yield "done", {"turn_id": turn_id, "usage": plan["usage"], "model": MODEL, "tool_calls_seen": plan["ncalls"]}

    def explain_text(self, job_id: str, action_id: str | None = None) -> str:
        """Explain an action of THIS job from its server-side log."""
        _job_dir, store = self.svc.job_context(job_id)
        with store.lock:
            actions = dict(store.actions)
            ops = list(store.ops.values())
        return explain_action(actions, ops, action_id)

    def _prescreen(self, name: str, raw: dict[str, Any], message: str, available: set[str]) -> tuple[dict[str, Any] | None, list[Event]]:
        """Validate the call before anything runs.

        Returns:
            ``(args, [])`` to continue, or ``(None, events)`` when the call is answered without executing.

        Raises:
            CtlError: 422 for an unknown tool.
            BadArgs: When the arguments are invalid.
        """
        if name not in ctl_tools.TOOLS:
            raise CtlError(422, "unknown_tool", "unknown tool")
        if name not in available:
            return None, [("text", {"delta": unavailable_text(name)})]
        if name == "edit_line" and isinstance(raw.get("text"), str):
            raw = {**raw, "text": restore_verbatim(message, raw["text"])}
        if name == "retime" and retime_out_of_scope(message):
            return None, [("text", {"delta": RETIME_REFUSAL})]
        return validate_args(name, raw), []

    def _answer_locally(self, job_id: str, name: str, args: dict[str, Any], snap: dict[str, Any]) -> list[Event] | None:
        """Answer tools that need no server action (client actions, explain); ``None`` means "submit an action"."""
        if name == "add_language" and args["lang"] in snap["langs"]:  # already dubbed: the user means "switch to it", never a costly add
            return [("client_action", {"tool": "switch_language", "args": {"lang": args["lang"]}})]
        if name == "explain":
            return text_events(self.explain_text(job_id, args.get("action_id")))
        if name in CLIENT_TOOLS:
            if name == "switch_language" and args["lang"] not in snap["langs"]:
                raise CtlError(422, "unknown_lang", "that language is not part of this job")
            return [("client_action", {"tool": name, "args": args})]
        return None

    def _build_body(
        self, job_id: str, turn_id: str, name: str, args: dict[str, Any], context: dict[str, Any], snap: dict[str, Any]
    ) -> dict[str, Any]:
        """Build the action request body for tool ``name``."""
        lang = context.get("lang") if context.get("lang") in snap["langs"] else snap["lang"]
        segment = resolve_segment(snap, args["segment_id"]) if "segment_id" in args else None
        seg_body: dict[str, Any] = {}
        if segment is not None:
            head = self.svc.head_view(job_id, lang, segment[0], segment[1])[0]
            seg_body = {"expected_revision": head["revision"]}
        return {
            "type": name,
            "args": submission_args(name, args, context, lang, segment),
            "idempotency_key": f"{turn_id}-{name}",
            "preview_only": name in PREVIEW_TOOLS,
            **seg_body,
        }

    @staticmethod
    def _result_events(name: str, args: dict[str, Any], turn_id: str, response: dict[str, Any]) -> list[Event]:
        """Turn the action service's answer into stream events."""
        if response.get("status") in ("needs_confirmation", "fork_proposal"):
            proposal = response["proposal"]
            return [
                (
                    "proposal",
                    {
                        "turn_id": turn_id, "tool": name, "args": proposal["args"], "summary": proposal["summary"],
                        "requires_confirmation": True, "expected_revision": proposal.get("expected_revision"),
                        "confirmation_token": response["confirmation_token"], "kind": response["status"],
                    },
                )
            ]
        events: list[Event] = [
            (
                "action_result",
                {"action_id": response.get("action_id"), "status": response.get("status"), "operation_id": (response.get("operation") or {}).get("operation_id")},
            )
        ]
        if name == "retime":  # never silent: state the tempo actually previewed and the supported range
            tempo = format_tempo(args["tempo_factor"])
            events.append(("text", {"delta": f" (Previewing tempo x{tempo} on that line; the supported range is 0.92 to 1.08, at most about 8 percent.)"}))
        return events

    async def route(
        self, job_id: str, turn_id: str, call: dict[str, Any], context: dict[str, Any], snap: dict[str, Any], available: set[str], message: str = ""
    ) -> AsyncIterator[Event]:
        """Validate and execute the planner's single tool call, yielding stream events."""
        name, raw = call["name"], call["args"]
        message = (message or "")[:1000]  # exactly what the planner saw: the guards must not read beyond it
        try:
            args, early = self._prescreen(name, raw, message, available)
            if args is None:
                for event in early:
                    yield event
                return
            local = self._answer_locally(job_id, name, args, snap)
            if local is not None:
                for event in local:
                    yield event
                return
            body = self._build_body(job_id, turn_id, name, args, context, snap)
            _status, response = await self.svc.submit(job_id, body)
            for event in self._result_events(name, args, turn_id, response):
                yield event
        except BadArgs as exc:
            yield "error", {"code": "invalid_args", "detail": str(exc)}
        except CtlError as exc:
            for event in error_events(exc, name):
                yield event
