"""Authentication and global admission control for the control-plane routes.

In-process state only: a single uvicorn worker is asserted by the deploy script.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sunwai.control.errors import CtlError
from sunwai.control.ratelimit import SlidingWindow
from sunwai.control.settings import env_float

JOB_RE = re.compile(r"^[0-9a-f]{12}$")
LANG_RE = re.compile(r"^[a-z]{2,3}$")
SEG_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
OP_RE = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")
COOKIE = "dub_ctl"
LOOPBACK_PEERS = ("127.0.0.1", "::1", "localhost")
MAX_TRACKED_ADDRESSES = 5000


def load_or_create_token(token_path: Path) -> str:
    """Return the control token: ``DUB_CTL_TOKEN``, else the file at ``token_path``, else a fresh mode-600 file."""
    token = os.environ.get("DUB_CTL_TOKEN", "").strip()
    if token:
        return token
    try:
        token = token_path.read_text().strip()
        if token:
            return token
    except OSError:
        pass
    token = secrets.token_urlsafe(24)
    fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(token)
    return token


def client_address(peer: str, forwarded: str | None) -> str:
    """Resolve the caller's address; ``cf-connecting-ip`` is honoured only when the TCP peer is the local proxy."""
    if peer in LOOPBACK_PEERS:
        return (forwarded or peer)[:64]
    return peer[:64]


class DailyLedger:
    """Global daily spend persisted to a JSON file; the totals reset when the UTC day changes."""

    def __init__(self, path: Path) -> None:
        """Bind the ledger to its state file."""
        self.path = path

    def load(self) -> dict[str, Any]:
        """Return today's totals (a fresh record when the file is missing, unreadable or from another day)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            state = json.loads(self.path.read_text())
        except Exception:
            state = {}
        if state.get("day") != today:
            state = {"day": today, "usd": 0.0, "synth_s": 0.0, "gemini_calls": 0}
        return state

    def save(self, state: dict[str, Any]) -> None:
        """Persist the totals atomically."""
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        os.replace(tmp, self.path)


class Guard:
    """Token auth, per-address rate limits, daily budget, Gemini gate, global synthesis slot and SSE caps."""

    def __init__(self, app_dir: str | Path, state_dir: str | Path) -> None:
        """Load the token from ``app_dir`` and bind the daily ledger under ``state_dir``."""
        self.lock = threading.RLock()
        self.token_path = Path(app_dir) / "ctl_token"
        self.token = load_or_create_token(self.token_path)
        self.ledger = DailyLedger(Path(state_dir) / "ctl_global.json")
        self.ip_hits: defaultdict[tuple[str, str], SlidingWindow] = defaultdict(SlidingWindow)
        self.gem = SlidingWindow(60.0)
        self.synth_running = 0
        self.sse_total = 0
        self.sse_job: defaultdict[str, int] = defaultdict(int)

    # ---- token
    def authed(self, request: Any) -> bool:
        """Tell whether the request carries the right token (header or cookie), compared in constant time."""
        got = request.headers.get("x-dub-token") or request.cookies.get(COOKIE) or ""
        return hmac.compare_digest(got.encode("utf-8", "replace"), self.token.encode())

    def require(self, request: Any) -> None:
        """Raise 401 unless :meth:`authed`."""
        if not self.authed(request):
            raise CtlError(401, "unauthorized", "missing or wrong control token")

    # ---- per address
    @staticmethod
    def ip_of(request: Any) -> str:
        """Return the caller's address for rate limiting."""
        peer = request.client.host if request.client else "?"
        return client_address(peer, request.headers.get("cf-connecting-ip"))

    def ip_limit(self, request: Any, kind: str, per_min: int) -> None:
        """Count a request of ``kind`` against the caller's per-minute allowance.

        Raises:
            CtlError: 429 ``rate_limited`` when the allowance is used up.
        """
        key = (self.ip_of(request), kind)
        with self.lock:
            retry = self.ip_hits[key].try_acquire(per_min)
            if retry is not None:
                raise CtlError(429, "rate_limited", f"too many {kind} requests from this address", retry_after=retry)
            if len(self.ip_hits) > MAX_TRACKED_ADDRESSES:  # bound memory
                idle = [k for k, window in self.ip_hits.items() if window.is_idle()]
                for stale in idle[: MAX_TRACKED_ADDRESSES // 2]:
                    self.ip_hits.pop(stale, None)

    # ---- global daily spend
    def check_daily(self, need_usd: float = 0.0, need_synth_s: float = 0.0) -> None:
        """Refuse work that would exceed today's global LLM or synthesis budget.

        Raises:
            CtlError: 429 ``daily_cap`` when a budget would be exceeded.
        """
        with self.lock:
            state = self.ledger.load()
            if state["usd"] + need_usd > env_float("DUB_CTL_DAILY_USD", 5.0):
                raise CtlError(429, "daily_cap", "daily LLM budget reached", retry_after=3600)
            if state["synth_s"] + need_synth_s > env_float("DUB_CTL_DAILY_SYNTH_S", 3600):
                raise CtlError(429, "daily_cap", "daily synthesis budget reached", retry_after=3600)

    def charge(self, usd: float = 0.0, synth_s: float = 0.0, gemini_calls: int = 0) -> None:
        """Charge an attempt to today's ledger, success or not."""
        with self.lock:
            state = self.ledger.load()
            state["usd"] = round(state["usd"] + usd, 6)
            state["synth_s"] = round(state["synth_s"] + synth_s, 3)
            state["gemini_calls"] = state.get("gemini_calls", 0) + gemini_calls
            self.ledger.save(state)

    def gemini_gate(self) -> None:
        """Count one planner call against the global per-minute allowance.

        Raises:
            CtlError: 429 ``rate_limited`` when the planner is saturated.
        """
        with self.lock:
            if self.gem.try_acquire(int(env_float("DUB_CTL_GEMINI_PER_MIN", 30))) is not None:
                raise CtlError(429, "rate_limited", "planner is busy, try again shortly", retry_after=10)

    # ---- global synthesis slot
    def synth_acquire(self) -> None:
        """Take a global synthesis slot.

        Raises:
            CtlError: 429 ``one_synthesis_global`` when all slots are taken.
        """
        with self.lock:
            if self.synth_running >= int(env_float("DUB_CTL_GLOBAL_SYNTH", 1)):
                raise CtlError(
                    429,
                    "one_synthesis_global",
                    "another synthesis is running; try again when it finishes",
                    retry_after=5,
                )
            self.synth_running += 1

    def synth_release(self) -> None:
        """Give a global synthesis slot back."""
        with self.lock:
            self.synth_running = max(0, self.synth_running - 1)

    # ---- SSE connections
    def sse_open(self, job_id: str) -> None:
        """Register an open event stream.

        Raises:
            CtlError: 429 ``too_many_streams`` when the global or per-job stream cap is reached.
        """
        with self.lock:
            too_many = self.sse_total >= int(env_float("DUB_CTL_SSE_TOTAL", 60))
            too_many = too_many or self.sse_job[job_id] >= int(env_float("DUB_CTL_SSE_PER_JOB", 6))
            if too_many:
                raise CtlError(429, "too_many_streams", "too many open event streams", retry_after=5)
            self.sse_total += 1
            self.sse_job[job_id] += 1

    def sse_close(self, job_id: str) -> None:
        """Unregister an event stream."""
        with self.lock:
            self.sse_total = max(0, self.sse_total - 1)
            self.sse_job[job_id] = max(0, self.sse_job[job_id] - 1)
