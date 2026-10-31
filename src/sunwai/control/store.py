"""append-only revision / action / operation / event store (per job). No server2 imports.
Files: <job>/ctl/{revisions,actions,operations,events}.jsonl ; all writes are append + fsync; latest record wins on load."""
import asyncio, hashlib, json, os, threading, time, uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path


EV_KEEP = int(os.environ.get("DUB_CTL_EV_KEEP", "2000"))
EV_REPLAY = int(os.environ.get("DUB_CTL_EV_REPLAY", "200"))
EV_MAX_BYTES = int(os.environ.get("DUB_CTL_EV_MAX_BYTES", str(4 << 20)))


def utcnow():
    return datetime.now(timezone.utc).isoformat()


class CtlError(Exception):
    def __init__(self, status, code, detail, **extra):
        super().__init__(detail)
        self.status, self.code, self.detail, self.extra = status, code, detail, extra


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def seg_key(lang, chunk, seg):
    return f"{lang}/{int(chunk)}/{seg}"


class JobStore:
    def __init__(self, job_dir):
        self.dir = Path(job_dir) / "ctl"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.revs = defaultdict(list)       # seg_key -> [rev dict] (append order)
        self.actions = {}                   # action_id -> record
        self.idem = {}                      # idempotency_key -> (payload_hash, action_id)
        self.ops = {}                       # operation_id -> latest record
        self.events = []                    # list of event dicts (seq == index+1)
        self.running = {}                   # seg_key/chunk key -> operation_id (in-flight, in memory)
        self.previews = deque()             # timestamps
        self.spend = {"usd": 0.0, "synth_s": 0.0}
        self.resumable = []
        self.lang_gen = {}                  # lang -> max playback_generation
        self._ev = None
        self._load()

    # ---- persistence
    def _p(self, name):
        return self.dir / f"{name}.jsonl"

    def _append(self, name, rec):
        line = canon(rec) + "\n"
        with open(self._p(name), "a", encoding="utf-8") as f:
            f.write(line); f.flush(); os.fsync(f.fileno())

    def _read(self, name):
        p = self._p(name)
        if not p.exists():
            return []
        out = []
        for ln in p.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(ln))
            except Exception:
                pass                       # torn last line after a crash is ignored
        return out

    def _read_tail(self, name, n):
        """Last n parseable lines without reading the whole file (seek from the end)."""
        p = self._p(name)
        if not p.exists():
            return []
        size = p.stat().st_size
        with open(p, "rb") as f:
            back = min(size, 1 << 20)
            f.seek(size - back)
            data = f.read()
        lines = data.splitlines()
        if back < size:
            lines = lines[1:]                              # first line may be cut
        out = []
        for ln in lines[-n:]:
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
        return out

    def _rotate_events(self):
        p = self._p("events")
        try:
            if p.exists() and p.stat().st_size > EV_MAX_BYTES:
                os.replace(p, self.dir / "events.jsonl.1")  # keep exactly one old generation; the live file restarts with the tail
                with open(p, "w", encoding="utf-8") as f:
                    for ev in self.events:
                        f.write(canon(ev) + "\n")
        except OSError:
            pass

    def _load(self):
        latest = {}
        for r in self._read("revisions"):
            k = seg_key(r["lang"], r["chunk_id"], r["segment_id"])
            latest[(k, r["revision"])] = r                  # later record for the same revision (status change) wins
        for (k, _n), r in sorted(latest.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            self.revs[k].append(r)
        for a in self._read("actions"):
            self.actions[a["action_id"]] = a
            if a.get("idempotency_key"):
                self.idem[a["idempotency_key"]] = (a["payload_hash"], a["action_id"])
        for o in self._read("operations"):
            self.ops[o["operation_id"]] = o
        self.events = self._read_tail("events", EV_KEEP)
        self.last_seq = self.events[-1]["seq"] if self.events else 0
        self._rotate_events()
        # restart recovery: operations that were queued/running when the process died can never finish
        self.resumable = []                  # add_language ops still within their deadline: re-attached by the service, NOT failed
        for o in list(self.ops.values()):
            if o.get("state") in ("queued", "running"):
                if o.get("kind") == "add_language" and o.get("stage") == "preparing" and float(o.get("deadline") or 0) > time.time():
                    self.resumable.append(dict(o)); continue
                o = dict(o); o.update(state="failed", stage="error", error="interrupted by server restart" if o.get("kind") != "add_language" else "language deadline passed while the server was down")
                self.put_op(o)
                self.emit("op.error", operation_id=o["operation_id"], action_id=o.get("action_id"), error=o["error"])
        sp = self.dir / "spend.json"
        if sp.exists():
            try:
                self.spend.update(json.loads(sp.read_text()))
            except Exception:
                pass

    # ---- revisions
    def head(self, key):
        """Active head = highest-numbered non-candidate, non-failed revision."""
        with self.lock:
            for r in reversed(self.revs.get(key, [])):
                if r.get("status") == "active":
                    return r
            return None

    def all_revs(self, key):
        with self.lock:
            return list(self.revs.get(key, []))

    def next_no(self, key):
        with self.lock:
            return max([r["revision"] for r in self.revs.get(key, [])] + [-1]) + 1

    def add_revision(self, key, rev, expect_head=None, check=False):
        """Append a revision. With check=True the CAS must hold: current active head revision == expect_head."""
        with self.lock:
            h = self.head(key)
            cur = h["revision"] if h else None
            if check and cur != expect_head:
                raise CtlError(409, "stale_revision", "segment changed since you loaded it", expected=expect_head, current=cur)
            rev = dict(rev)
            rev["revision"] = self.next_no(key)
            self._append("revisions", rev)
            self.revs[key].append(rev)
            return rev

    def set_status(self, key, revno, status):
        """Status changes are NEW appended records (same revision number, later ts); latest wins."""
        with self.lock:
            lst = self.revs[key]
            for i in range(len(lst) - 1, -1, -1):
                if lst[i]["revision"] == revno:
                    nr = dict(lst[i]); nr["status"] = status; nr["ts_status"] = utcnow()
                    self._append("revisions", nr)
                    lst[i] = nr
                    return nr
        return None

    # ---- actions / ops
    def idem_lookup(self, idem_key, payload_hash):
        with self.lock:
            hit = self.idem.get(idem_key)
            if not hit:
                return None
            if hit[0] != payload_hash:
                raise CtlError(422, "idempotency_conflict", "idempotency_key reused with a different payload")
            return self.actions.get(hit[1])

    def idem_reserve(self, idem_key, payload_hash, action_id):
        """Atomic lookup-or-reserve. -> (existing_record_or_None, in_progress_action_id_or_None). Caller must idem_release on failure."""
        with self.lock:
            hit = self.idem.get(idem_key)
            if hit:
                if hit[0] != payload_hash:
                    raise CtlError(422, "idempotency_conflict", "idempotency_key reused with a different payload")
                return self.actions.get(hit[1]), hit[1]
            self.idem[idem_key] = (payload_hash, action_id)
            return None, None

    def idem_release(self, idem_key, action_id):
        with self.lock:
            hit = self.idem.get(idem_key)
            if hit and hit[1] == action_id and action_id not in self.actions:
                self.idem.pop(idem_key, None)

    def add_action(self, rec):
        with self.lock:
            self._append("actions", rec)
            self.actions[rec["action_id"]] = rec
            if rec.get("idempotency_key"):
                self.idem[rec["idempotency_key"]] = (rec["payload_hash"], rec["action_id"])

    def put_op(self, op):
        with self.lock:
            op = dict(op); op["ts"] = utcnow()
            self._append("operations", op)
            self.ops[op["operation_id"]] = op
            return op

    # ---- events (SSE)
    def emit(self, typ, **data):
        with self.lock:
            self.last_seq += 1
            ev = {"seq": self.last_seq, "ts": utcnow(), "type": typ, "job_id": self.dir.parent.name}
            for k_ in ("seq", "ts", "type"):
                data.pop(k_, None)                         # the envelope fields can never be overwritten by a payload kwarg
            ev.update(data)
            self._append("events", ev)
            self.events.append(ev)
            if len(self.events) > EV_KEEP * 2:
                del self.events[:-EV_KEEP]
                self._rotate_events()
            old, self._ev = self._ev, None
        if old is not None:
            try:
                old[1].call_soon_threadsafe(old[0].set)
            except Exception:
                pass
        return ev

    def waiter(self):
        """(Event, loop) for the next append; created lazily by an SSE consumer inside the loop."""
        with self.lock:
            if self._ev is None:
                self._ev = (asyncio.Event(), asyncio.get_running_loop())
            return self._ev[0]

    def events_after(self, after, limit=None):
        """-> (events with seq > after (at most the LAST `limit`), reset_needed). reset_needed when the client is behind what we retain."""
        limit = limit or EV_REPLAY
        with self.lock:
            after = max(0, after)
            evs = [e for e in self.events if e["seq"] > after]
            reset = len(evs) > limit or bool(self.events and self.events[0]["seq"] > after + 1)
            if len(evs) > limit:
                evs = evs[-limit:]
            return evs, reset

    # ---- limits / spend
    def check_preview_rate(self, per_min):
        now = time.time()
        with self.lock:
            while self.previews and now - self.previews[0] > 60:
                self.previews.popleft()
            if len(self.previews) >= per_min:
                raise CtlError(429, "rate_limited", f"at most {per_min} previews per minute", retry_after=int(60 - (now - self.previews[0])) + 1)
            self.previews.append(now)

    def add_spend(self, usd=0.0, synth_s=0.0):
        with self.lock:
            self.spend["usd"] = round(self.spend["usd"] + usd, 6)
            self.spend["synth_s"] = round(self.spend["synth_s"] + synth_s, 3)
            tmp = self.dir / "spend.json.tmp"
            tmp.write_text(json.dumps(self.spend)); os.replace(tmp, self.dir / "spend.json")

    def check_caps(self, usd_cap, synth_cap_s, need_synth_s=0.0):
        with self.lock:
            if self.spend["usd"] >= usd_cap:
                raise CtlError(429, "spend_cap", "per-job LLM spend cap reached", retry_after=3600)
            if self.spend["synth_s"] + need_synth_s > synth_cap_s:
                raise CtlError(429, "spend_cap", "per-job synthesis budget reached", retry_after=3600)


_STORES = {}
_SLOCK = threading.Lock()


def store_for(job_dir):
    k = str(Path(job_dir).resolve())
    with _SLOCK:
        if k not in _STORES:
            _STORES[k] = JobStore(job_dir)
        return _STORES[k]
