"""auth + global admission for the NEW routes. In-process (single uvicorn worker is asserted by the deploy script)."""
import json, os, re, secrets, threading, time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

from sunwai.control.store import CtlError

JOB_RE = re.compile(r"^[0-9a-f]{12}$")
LANG_RE = re.compile(r"^[a-z]{2,3}$")
SEG_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
OP_RE = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")
COOKIE = "dub_ctl"


def _f(n, d):
    return float(os.environ.get(n, d))


class Guard:
    def __init__(self, app_dir, state_dir):
        self.lock = threading.RLock()
        self.token_path = Path(app_dir) / "ctl_token"
        self.token = self._load_token()
        self.state_path = Path(state_dir) / "ctl_global.json"
        self.ip_hits = defaultdict(deque)
        self.gem = deque()
        self.synth_running = 0
        self.sse_total = 0
        self.sse_job = defaultdict(int)

    # ---- token (file ctl_token, mode 600; created if missing; value never logged)
    def _load_token(self):
        t = os.environ.get("DUB_CTL_TOKEN", "").strip()
        if t:
            return t
        try:
            t = self.token_path.read_text().strip()
            if t:
                return t
        except OSError:
            pass
        t = secrets.token_urlsafe(24)
        fd = os.open(self.token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(t)
        return t

    def authed(self, request):
        import hmac
        got = request.headers.get("x-dub-token") or request.cookies.get(COOKIE) or ""
        return hmac.compare_digest(got.encode("utf-8", "replace"), self.token.encode())

    def require(self, request):
        if not self.authed(request):
            raise CtlError(401, "unauthorized", "missing or wrong control token")

    # ---- per-IP
    @staticmethod
    def ip_of(request):
        peer = request.client.host if request.client else "?"
        # cf-connecting-ip is honoured ONLY when the TCP peer is the local cloudflared (loopback); otherwise it is attacker-controlled
        if peer in ("127.0.0.1", "::1", "localhost"):
            return (request.headers.get("cf-connecting-ip") or peer)[:64]
        return peer[:64]

    def ip_limit(self, request, kind, per_min):
        k = (self.ip_of(request), kind)
        now = time.time()
        with self.lock:
            q = self.ip_hits[k]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= per_min:
                raise CtlError(429, "rate_limited", f"too many {kind} requests from this address", retry_after=int(60 - (now - q[0])) + 1)
            q.append(now)
            if len(self.ip_hits) > 5000:                       # bound memory
                for kk in [kk for kk, v in self.ip_hits.items() if not v or now - v[-1] > 60][:2500]:
                    self.ip_hits.pop(kk, None)

    # ---- global daily spend (persisted across restarts)
    def _state(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            d = json.loads(self.state_path.read_text())
        except Exception:
            d = {}
        if d.get("day") != today:
            d = {"day": today, "usd": 0.0, "synth_s": 0.0, "gemini_calls": 0}
        return d

    def _save(self, d):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d)); os.replace(tmp, self.state_path)

    def check_daily(self, need_usd=0.0, need_synth_s=0.0):
        with self.lock:
            d = self._state()
            if d["usd"] + need_usd > _f("DUB_CTL_DAILY_USD", 5.0):
                raise CtlError(429, "daily_cap", "daily LLM budget reached", retry_after=3600)
            if d["synth_s"] + need_synth_s > _f("DUB_CTL_DAILY_SYNTH_S", 3600):
                raise CtlError(429, "daily_cap", "daily synthesis budget reached", retry_after=3600)

    def charge(self, usd=0.0, synth_s=0.0, gemini_calls=0):
        """Charged on every attempt, success or not."""
        with self.lock:
            d = self._state()
            d["usd"] = round(d["usd"] + usd, 6); d["synth_s"] = round(d["synth_s"] + synth_s, 3); d["gemini_calls"] = d.get("gemini_calls", 0) + gemini_calls
            self._save(d)

    def gemini_gate(self):
        """Global Gemini calls per minute; counts every attempt."""
        now = time.time(); n = int(_f("DUB_CTL_GEMINI_PER_MIN", 30))
        with self.lock:
            while self.gem and now - self.gem[0] > 60:
                self.gem.popleft()
            if len(self.gem) >= n:
                raise CtlError(429, "rate_limited", "planner is busy, try again shortly", retry_after=10)
            self.gem.append(now)

    # ---- global synthesis slot
    def synth_acquire(self):
        with self.lock:
            if self.synth_running >= int(_f("DUB_CTL_GLOBAL_SYNTH", 1)):
                raise CtlError(429, "one_synthesis_global", "another synthesis is running; try again when it finishes", retry_after=5)
            self.synth_running += 1

    def synth_release(self):
        with self.lock:
            self.synth_running = max(0, self.synth_running - 1)

    # ---- SSE connections
    def sse_open(self, job_id):
        with self.lock:
            if self.sse_total >= int(_f("DUB_CTL_SSE_TOTAL", 60)) or self.sse_job[job_id] >= int(_f("DUB_CTL_SSE_PER_JOB", 6)):
                raise CtlError(429, "too_many_streams", "too many open event streams", retry_after=5)
            self.sse_total += 1; self.sse_job[job_id] += 1

    def sse_close(self, job_id):
        with self.lock:
            self.sse_total = max(0, self.sse_total - 1)
            self.sse_job[job_id] = max(0, self.sse_job[job_id] - 1)
