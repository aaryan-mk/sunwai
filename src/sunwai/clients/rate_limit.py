"""Hosted-API rate limiting: a cross-process sliding-window gate and Retry-After parsing.

Observed on the hosted Bodhan API: TTS answers 429 with ``Retry-After`` after about 8 requests in a minute; MT allows
about 120 requests per minute and 10 parallel requests per key. Only used when ``BODHAN_MODE=api``.
"""

from __future__ import annotations

import datetime as _dt
import fcntl
import json
import logging
import os
import random as _random
import tempfile
import threading as _threading
import time
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Dict

import requests

from sunwai.clients import bodhan_endpoints as bodhan_cfg
from sunwai.pipeline.support.env import _env_num

log = logging.getLogger(__name__)

API_RPM_DEFAULT = {"asr": 60, "mt": 100, "tts": 8}


def api_mode() -> bool:
    """True when ``BODHAN_MODE`` selects the hosted API."""
    return bodhan_cfg.mode() == "api"


def parse_retry_after(value: Any, cap: float = 300.0) -> float | None:
    """Parse ``Retry-After`` (seconds or HTTP-date) into ``[0, cap]`` seconds; ``None`` if absent or invalid."""
    if value is None:
        return None
    v = str(value).strip()
    if not v:
        return None
    try:
        sec = float(v)
    except ValueError:
        try:
            d = parsedate_to_datetime(v)
            sec = (d - _dt.datetime.now(d.tzinfo or _dt.timezone.utc)).total_seconds()
        except Exception:
            return None
    if sec != sec or sec == float("inf"):
        return None
    return max(0.0, min(float(cap), sec))


class RateGate:
    """Cross-process sliding-window requests-per-minute gate.

    State lives in ``<slot_dir>/<stage>.rate.json`` under ``flock``. ``acquire`` reserves a request slot (waiting for
    it); ``block`` records a server-told ``Retry-After`` so every process pauses. Fails open on ``OSError``.
    """

    def __init__(self, stage: str, rpm: int, slot_dir: str | None = None) -> None:
        self.stage, self.rpm = stage, max(1, int(rpm))
        self.dir = slot_dir or os.environ.get("DUB_SLOT_DIR", os.path.join(tempfile.gettempdir(), "sunwai_slots"))
        self._broken = os.environ.get("DUB_SLOTS", "1") == "0"

    def _txn(self, fn: Callable[[Dict[str, Any]], Any]) -> Any:
        os.makedirs(self.dir, exist_ok=True)
        path = os.path.join(self.dir, f"{self.stage}.rate.json")
        fd = os.open(path + ".lock", os.O_CREAT | os.O_RDWR, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                with open(path) as f:
                    st = json.load(f)
                if not isinstance(st, dict):
                    st = {}
            except Exception:
                st = {}
            res = fn(st)
            tmp = f"{path}.tmp{os.getpid()}_{_threading.get_ident()}"
            with open(tmp, "w") as f:
                json.dump(st, f)
            os.replace(tmp, path)
            return res
        finally:
            os.close(fd)

    def acquire(self, max_wait: float) -> float:
        """Reserve a request slot, return seconds waited; raise Timeout if that would exceed ``max_wait``."""
        if self._broken:
            return 0.0
        t0 = time.time()
        t_end = t0 + max(0.0, max_wait)
        while True:

            def fn(st: Dict[str, Any]) -> float:
                now = time.time()
                ts = [t for t in st.get("ts", []) if now - t < 60.0]
                wait = max(0.0, st.get("blocked_until", 0.0) - now)
                if wait <= 0.0 and len(ts) < self.rpm:
                    ts.append(now)
                    st["ts"] = ts
                    return 0.0
                if wait <= 0.0:
                    wait = ts[len(ts) - self.rpm] + 60.0 - now
                st["ts"] = ts
                return float(max(0.05, wait))

            try:
                wait = self._txn(fn)
            except OSError as e:
                log.warning("rate gate for %s disabled (%s)", self.stage, type(e).__name__)
                self._broken = True
                return 0.0
            if wait <= 0.0:
                return float(time.time() - t0)
            if time.time() + wait > t_end:
                raise requests.exceptions.Timeout(
                    f"{self.stage} rate limit ({self.rpm}/min) would need {wait:.0f}s > budget"
                )
            time.sleep(wait + _random.uniform(0.0, 0.2))

    def block(self, seconds: float) -> None:
        """Pause every process for ``seconds`` (a server ``Retry-After``)."""
        if self._broken or not seconds:
            return

        def fn(st: Dict[str, Any]) -> None:
            st["blocked_until"] = max(st.get("blocked_until", 0.0), time.time() + float(seconds))

        try:
            self._txn(fn)
        except OSError:
            pass


_GATES: Dict[str, RateGate] = {}


def rate_gate(stage: str) -> RateGate | None:
    """Return the stage's gate in api mode, else ``None``. Limit: ``BODHAN_API_RPM_<STAGE>``."""
    if not api_mode():
        return None
    g = _GATES.get(stage)
    if g is None:
        rpm = _env_num(  # type: ignore[no-untyped-call]
            f"BODHAN_API_RPM_{stage.upper()}", API_RPM_DEFAULT[stage], int
        )
        g = _GATES.setdefault(stage, RateGate(stage, rpm))
    return g


def api_rate_wait() -> float:
    """Longest a call waits for the rate window (``BODHAN_API_RATE_WAIT``, default 300 s)."""
    return float(
        _env_num(  # type: ignore[no-untyped-call]
            "BODHAN_API_RATE_WAIT", 300.0, float
        )
    )
