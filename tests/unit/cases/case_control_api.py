"""Anonymous GET /api/ctl/config -> 200 {"enabled": false} only; authed unchanged; other ctl endpoints still 401. ."""

import json, os, sys, tempfile
from pathlib import Path

os.environ.update(DUB_ACTIONS="1", DUB_CTL_TOKEN="testtok", DUB_CTL_DRYRUN="1", DUB_CTL_IP_READ_PER_MIN="1000")
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sunwai.server.routers import control as ctl_api
import asyncio

_LOCKS = {}


def _get_lock(key):
    return _LOCKS.setdefault(key, asyncio.Lock())


ROOT = Path(tempfile.mkdtemp(prefix="ctl_api_"))
JID = "0123456789ab"
(ROOT / JID / "hi" / "c0000").mkdir(parents=True)
(ROOT / JID / "job.json").write_text(json.dumps({"job_id": JID, "langs": ["hi"]}))
ns = {
    "resolve_job": lambda j: (ROOT / j) if j == JID else None,
    "read_json": lambda p: {},
    "resolve_chunk_dir": lambda jd, l, c: jd / l / f"c{c:04d}",
    "_load_segments": lambda od: [],
    "_save_segments": lambda od, s: None,
    "_find_segment": lambda s, i: (-1, None),
    "_status_path": lambda od: od / "status.json",
    "get_target_codes": lambda: {"hi"},
    "ensure_job_running": lambda j: "running",
    "MAX_TARGETS": 3,
    "get_lock": _get_lock,
    "APP_DIR": ROOT,
    "DUB_ROOT": ROOT,
}
app = FastAPI()
assert ctl_api.register(app, ns)
c = TestClient(app)
fails = []


def check(n, ok):
    print(("PASS " if ok else "FAIL ") + n)
    if not ok:
        fails.append(n)


r = c.get("/api/ctl/config")
check("anonymous config -> 200", r.status_code == 200)
check('anonymous body is exactly {"enabled": false}', r.status_code == 200 and r.json() == {"enabled": False})
r = c.get("/api/ctl/config", headers={"x-dub-token": "wrong"})
check("wrong token -> no disclosure", r.status_code in (200, 401) and "tools" not in r.text and "limits" not in r.text)
r = c.get("/api/ctl/config", headers={"x-dub-token": "testtok"})
j = r.json() if r.status_code == 200 else {}
check(
    "authed config unchanged (enabled true + tools + limits)",
    r.status_code == 200 and j.get("enabled") is True and "tools" in j and "limits" in j,
)
r = c.get("/api/ctl/config", cookies={"dub_ctl": "testtok"})
check("cookie auth unchanged", r.status_code == 200 and r.json().get("enabled") is True)
check("anonymous POST actions still 401", c.post(f"/api/jobs/{JID}/actions", json={}).status_code == 401)
check("anonymous POST languages still 401", c.post(f"/api/jobs/{JID}/languages", json={}).status_code == 401)
check("anonymous GET events still 401", c.get(f"/api/jobs/{JID}/events").status_code == 401)
check("anonymous POST chat still 401", c.post(f"/api/jobs/{JID}/chat", json={}).status_code == 401)
os.environ["DUB_CTL_IP_READ_PER_MIN"] = "1000"
print("FAILS:", fails)
sys.exit(1 if fails else 0)
