"""Cross-process service slots, thread-local HTTP sessions and fault injection."""

from __future__ import annotations

import json
import logging
import os
import random as _random
import threading as _threading
import time

import requests

log = logging.getLogger(__name__)

from sunwai.clients.rate_limit import api_mode, api_rate_wait, rate_gate
from sunwai.clients.service_endpoints import MT_BASES, TTS_BASES
from sunwai.pipeline.support.env import _env_num


_TLS = _threading.local()


class _SlotLimiter:
    """Cross-process counting semaphore. Every chunk is its own subprocess with its own thread pools, so
    per-process worker counts multiply by the number of concurrent chunks and requests pile up on a service without
    bound. A stage may hold at most `limit` in-flight calls across ALL processes on this host: `limit` lock files, each
    flock'ed while a call is in flight (kernel-released if the process dies, so a crash never leaks a slot).
    - ONE config per host: <slot_dir>/limits.json ({"tts": 40, ...}) overrides the env/default limit, so processes started
      with different environments still enforce the same cap (the first user writes the file from its own env/default).
    - FIFO fairness: a waiter takes a ticket (<stage>.q.<time_ns>.<pid>.<tid>, flock'ed while waiting) and only the
      smallest LIVE ticket polls for a slot, so an older request is never overtaken (dead tickets are reaped).
    - Waiting raises requests.exceptions.Timeout after `wait_s` so _with_retries accounts for the wait.
    - Fail-open: if the slot directory, lock file or flock is unusable, calls proceed unbounded (logged once).
    flock is host-local semantics on some filesystems; all chunk processes run on one host."""

    def __init__(self, stage, limit, slot_dir=None, wait_s=None):
        self.stage = stage
        self.dir = slot_dir or os.environ.get("DUB_SLOT_DIR", os.path.join(__import__("tempfile").gettempdir(), "sunwai_slots"))
        self._req_limit = max(1, int(limit))
        self.limit = self._req_limit
        self.wait_s = _env_num("DUB_SLOT_WAIT", 120.0, float) if wait_s is None else wait_s
        self._held = set()
        self._mu = _threading.Lock()
        self._broken = os.environ.get("DUB_SLOTS", "1") == "0"
        self._resolved = False

    def _fail_open(self, why):
        if not self._broken:
            log.warning("slot limiter for %s disabled (%s): calls are unbounded", self.stage, why)
        self._broken = True

    def _resolve_limit(self):
        """Host-wide limit from limits.json; create it (atomically, first writer wins) from this process's value."""
        if self._resolved:
            return
        path = os.path.join(self.dir, "limits.json")
        import fcntl
        lfd = None
        try:
            lfd = os.open(os.path.join(self.dir, "limits.lock"), os.O_CREAT | os.O_RDWR, 0o666)
            fcntl.flock(lfd, fcntl.LOCK_EX)           # merge under the lock: every stage gets ONE host-wide value
            try:
                cfg = json.load(open(path))
            except Exception:
                cfg = None
            if not isinstance(cfg, dict):
                cfg = {}
            if self.stage not in cfg:                 # first writer for THIS stage wins; never overwrite
                cfg[self.stage] = self._req_limit
                tmp = f"{path}.tmp{os.getpid()}_{_threading.get_ident()}"
                with open(tmp, "w") as _f:
                    json.dump(cfg, _f)
                os.replace(tmp, path)
        except OSError:
            cfg = cfg if isinstance(locals().get("cfg"), dict) else {}
        finally:
            if lfd is not None:
                os.close(lfd)
        try:
            self.limit = max(1, int(cfg.get(self.stage, self._req_limit)))
        except (TypeError, ValueError):
            self.limit = self._req_limit
        self._resolved = True

    def _head(self, mine):
        """True if no older LIVE ticket exists; dead tickets are removed."""
        import fcntl
        try:
            names = sorted(n for n in os.listdir(self.dir) if n.startswith(f"{self.stage}.q."))
        except OSError:
            return True
        for n in names:
            if n >= mine:
                break
            fd = None
            try:
                fd = os.open(os.path.join(self.dir, n), os.O_RDWR)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:                       # lockable = owner is gone
                    os.remove(os.path.join(self.dir, n))
                except OSError:
                    pass
            except BlockingIOError:
                return False               # an older waiter is alive
            except OSError:
                pass
            finally:
                if fd is not None:
                    os.close(fd)
        return True

    def _try_slots(self):
        import fcntl
        for i in range(self.limit):
            with self._mu:
                if i in self._held:
                    continue
                self._held.add(i)          # in-process claim first (flock may not conflict inside one process)
            fd = None
            try:
                fd = os.open(os.path.join(self.dir, f"{self.stage}.{i}.lock"), os.O_CREAT | os.O_RDWR, 0o666)
            except OSError as e:           # cannot even open the lock file (permissions, read-only fs): fail open
                with self._mu:
                    self._held.discard(i)
                self._fail_open(e)
                return "open"
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return (i, fd)
            except BlockingIOError:
                pass                       # another process holds slot i
            except OSError as e:
                import errno
                if e.errno not in (errno.EACCES, errno.EAGAIN):
                    os.close(fd)
                    with self._mu:
                        self._held.discard(i)
                    self._fail_open(e)
                    return "open"
            os.close(fd)
            with self._mu:
                self._held.discard(i)
        return None

    def acquire(self, wait=None):
        """-> (index, fd) or None when fail-open."""
        import fcntl
        if self._broken:
            return None
        try:
            os.makedirs(self.dir, exist_ok=True)
            self._resolve_limit()
        except OSError as e:
            self._fail_open(e)
            return None
        wait = self.wait_s if wait is None else min(self.wait_s, wait)
        t_end = time.time() + wait
        tname = f"{self.stage}.q.{time.time_ns():020d}.{os.getpid()}.{_threading.get_ident()}"
        tfd = None
        try:
            try:
                tfd = os.open(os.path.join(self.dir, tname), os.O_CREAT | os.O_RDWR, 0o666)
                fcntl.flock(tfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                self._fail_open(e)
                return None
            while True:
                if self._head(tname):
                    got = self._try_slots()
                    if got == "open":
                        return None
                    if got:
                        return got
                if time.time() >= t_end:
                    raise requests.exceptions.Timeout(f"no {self.stage} slot free after {wait:.0f}s ({self.limit} in flight)")
                time.sleep(_random.uniform(0.01, 0.04))
        finally:
            if tfd is not None:
                try:
                    os.remove(os.path.join(self.dir, tname))
                except OSError:
                    pass
                os.close(tfd)

    def release(self, tok):
        import fcntl
        if tok is None:
            return
        i, fd = tok
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
            with self._mu:
                self._held.discard(i)

    def __call__(self, wait=None):
        lim = self

        class _Ctx:
            def __enter__(self_):
                self_.tok = lim.acquire(wait)
                return self_

            def __exit__(self_, *a):
                lim.release(self_.tok)
        return _Ctx()


_LIMITERS = {}


class _Gated:
    """A slot context that first waits for the stage's hosted rate window."""

    def __init__(self, inner, gate, rate_wait):
        self.inner, self.gate, self.rate_wait = inner, gate, rate_wait

    def __enter__(self):
        self.gate.acquire(self.rate_wait)  # raises requests Timeout when the budget cannot cover the rate window
        return self.inner.__enter__()

    def __exit__(self, *a):
        return self.inner.__exit__(*a)


def inflight(stage, wait=None, rate_wait=None):
    """`with inflight("tts"):` around ONE service call. Caps (env, across all chunk processes): ASR_MAX_INFLIGHT=8,
    MT_MAX_INFLIGHT=4 x replicas, TTS_MAX_INFLIGHT=4 x replicas; DUB_SLOTS=0 disables. In api mode the default is
    BODHAN_API_MAX_INFLIGHT=8 and each call first passes the stage's requests-per-minute gate (waiting up to ``rate_wait``)."""
    lim = _LIMITERS.get(stage)
    if lim is None:
        default = {"asr": 8, "mt": 4 * max(1, len(MT_BASES)), "tts": 4 * max(1, len(TTS_BASES))}[stage]
        if api_mode():  # hosted: Max_parallel_requests=10 per API key -> stay at <= 8 in flight per service
            default = _env_num("BODHAN_API_MAX_INFLIGHT", 8, int)
        lim = _LIMITERS.setdefault(stage, _SlotLimiter(stage, _env_num(f"{stage.upper()}_MAX_INFLIGHT", default, int)))
    gate = rate_gate(stage)
    if gate is not None:
        return _Gated(lim(wait), gate, api_rate_wait() if rate_wait is None else rate_wait)
    return lim(wait)


def thread_session() -> requests.Session:
    """One requests.Session per thread (Session sharing across threads is not guaranteed thread-safe)."""
    ss = getattr(_TLS, "session", None)
    if ss is None:
        ss = requests.Session()
        ss.headers.update({"User-Agent": "dub-pipeline/1.0"})
        _TLS.session = ss
    return ss


def _fault(stage: str, seg_id) -> None:
    """Fault-injection hook for tests: DUB_FAULT="asr:3,mt:4,tts:5" makes that stage raise for that segment id."""
    spec = os.environ.get("DUB_FAULT", "")
    if spec and f"{stage}:{seg_id}" in [x.strip() for x in spec.split(",")]:
        raise RuntimeError(f"injected {stage} fault for seg {seg_id}")
