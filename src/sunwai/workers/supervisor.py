#!/usr/bin/env python3
"""SunwAI supervisor (stdlib only). Runs in the foreground of one process or scheduler step.
 * keeps uvicorn (sunwai.server.app:create_app) alive: restarts only uvicorn, never the tunnel, so the public URL does not rotate
 * keeps the quick tunnel alive: adopts an existing cloudflared, restarts it only after the PUBLIC url failed 3x in a row,
   and then writes the new URL to tunnel_url.txt (+ tunnel_url_history.txt)
 * probes every dependency each cycle (ASR, MT x5, IT2, TTS x10, sep, both lip workers) and writes
   <ROOT>/supervisor_status.json atomically; logs every up/down transition.  `--once` prints the status and exits.
env: SUP_ROOT SUP_PORT SUP_NO_TUNNEL=1 SUP_APPDIR SUP_CLOUDFLARED
"""
import json, os, signal, subprocess, sys, time, urllib.request, re

ROOT = os.environ.get("SUP_ROOT", os.environ.get("DUB_ROOT", "./jobs"))
PORT = int(os.environ.get("SUP_PORT", "18941"))
APPDIR = os.environ.get("SUP_APPDIR", ".")
CF = os.environ.get("SUP_CLOUDFLARED", "cloudflared")
NO_TUNNEL = os.environ.get("SUP_NO_TUNNEL") == "1"
UV_PAT = f"uvicorn --factory sunwai.server.app:create_app --host 127.0.0.1 --port {PORT}"
CF_PAT = f"cloudflared tunnel --no-autoupdate --url http://127.0.0.1:{PORT}"
def log(*a):
    print(time.strftime("%F %T"), *a, flush=True)


def load_services():
    """{name: ["host:port",...]} from env SUP_SERVICES (JSON file, same format as the app's DUB_SERVICES) else
    DEP_HOST (default 127.0.0.1) + built-in ports. No node is hard-coded."""
    host = os.environ.get("DEP_HOST", "127.0.0.1")
    ports = {"asr": [18931], "mt": [18933, 18981, 18982, 18983, 18984], "it2": [18936],
             "tts": list(range(18990, 19000)), "sep": [18952]}
    cfg = {k: [f"{host}:{p}" for p in v] for k, v in ports.items()}
    lipm = os.environ.get("LIP_MODE", "proc")
    if lipm not in ("proc", "none"):
        lipm = "proc"
    lipw = [w.strip() for w in os.environ.get("LIP_WORKERS", "wA,wB").split(",") if w.strip()] or ["wA", "wB"]
    f = os.environ.get("SUP_SERVICES", os.environ.get("DUB_SERVICES", ""))
    if f:
        try:
            j = json.load(open(f))
            for k, v in j.items():
                if isinstance(v, list) and k != "lip_workers":
                    if v and all(":" in str(x) for x in v):
                        cfg[k] = [str(x) for x in v]
                    else:
                        log(f"ignoring empty/invalid list for {k}")
            lm, lw = j.get("lip_mode", lipm), j.get("lip_workers", lipw)
            if lm in ("proc", "none"):
                lipm = lm
            if isinstance(lw, list) and lw and all(isinstance(x, str) and x for x in lw):
                lipw = lw
        except Exception as e:
            log(f"cannot read services file {f}: {e!r}")
    return cfg, lipm, lipw


DEPS, LIP_MODE, LIP = load_services()
CYCLE = 5


def atomic(path, text):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def http_ok(url, timeout=3.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def pids_matching(pat):
    out = []
    me = os.getpid()
    for p in os.listdir("/proc"):
        if p.isdigit() and int(p) != me:
            try:
                cl = open(f"/proc/{p}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
            except OSError:
                continue
            if pat in cl and "supervisor_app2" not in cl:
                out.append(int(p))
    return out


def kill_all(pids, grace=8):
    for p in pids:
        try: os.kill(p, signal.SIGTERM)
        except OSError: pass
    t0 = time.time()
    while time.time() - t0 < grace and any(os.path.exists(f"/proc/{p}") for p in pids):
        time.sleep(0.3)
    for p in pids:
        if os.path.exists(f"/proc/{p}"):
            try: os.kill(p, signal.SIGKILL)
            except OSError: pass


def probe_deps():
    res = {}
    for n, ports in DEPS.items():
        down = [p for p in ports if not http_ok(f"http://{p}/health", 1.5)]
        res[n] = {"up": len(ports) - len(down), "total": len(ports), "down": down}
    lip = {}
    if LIP_MODE != "proc":
        return res, {w: -1 for w in LIP}      # -1 = not checkable from this host
    def cl(p):
        try:
            return open(f"/proc/{p}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace") + " "
        except OSError:
            return ""
    cands = pids_matching("sunwai.workers.lipsync")
    for w in LIP:
        ps = [p for p in cands if f"--id {w} " in cl(p)]
        lip[w] = ps[0] if ps else None
    return res, lip


def spawn_uvicorn():
    py = sys.executable
    env = dict(os.environ, DUB_ROOT=ROOT)
    lf = open(f"{ROOT}/server.log", "ab")
    p = subprocess.Popen([py, "-m", "uvicorn", "--factory", "sunwai.server.app:create_app", "--host", "127.0.0.1", "--port", str(PORT)],
                         cwd=APPDIR, env=env, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    lf.close()
    log(f"uvicorn spawned pid={p.pid}")
    return p


def spawn_tunnel():
    tl = f"{ROOT}/tunnel.log"
    open(tl, "w").close()
    lf = open(tl, "ab")
    p = subprocess.Popen([CF, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"],
                         stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    lf.close()
    url = ""
    for _ in range(40):
        time.sleep(2)
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open(tl, errors="replace").read())
        if m:
            url = m.group(0); break
    log(f"tunnel spawned pid={p.pid} url={url}")
    if url:
        atomic(f"{ROOT}/tunnel_url.txt", url + "\n")
        with open(f"{ROOT}/tunnel_url_history.txt", "a") as f:
            f.write(f"{time.strftime('%F %T')} {url}\n")
    return url


def read_url():
    try:
        return open(f"{ROOT}/tunnel_url.txt").read().strip()
    except OSError:
        return ""


def main():
    os.makedirs(ROOT, exist_ok=True)
    if "--once" in sys.argv:
        d, l = probe_deps()
        print(json.dumps({"deps": d, "lip": l, "uvicorn": pids_matching(UV_PAT), "tunnel": pids_matching(CF_PAT),
                          "url": read_url()}, indent=1))
        return
    log(f"supervisor start pid={os.getpid()} root={ROOT} port={PORT} tunnel={'off' if NO_TUNNEL else 'on'}")
    uv_fail = tn_fail = 0
    uv_grace = 0.0
    last_tn = 0.0
    prev_down = set()
    while True:
        t = time.time()
        # --- uvicorn
        uvp = pids_matching(UV_PAT)
        healthy = http_ok(f"http://127.0.0.1:{PORT}/health", 4.0)
        if healthy:
            uv_fail = 0
        elif not uvp:
            log("uvicorn not running -> start")
            spawn_uvicorn(); uv_grace = t + 60; uv_fail = 0
        elif t > uv_grace:
            uv_fail += 1
            if uv_fail >= 3:
                log(f"uvicorn unhealthy x{uv_fail} -> restart (pids {uvp})")
                kill_all(uvp); spawn_uvicorn(); uv_grace = time.time() + 60; uv_fail = 0
        # --- tunnel (public url checked every 30 s)
        if not NO_TUNNEL and t - last_tn >= 30:
            last_tn = t
            url = read_url()
            if url and http_ok(url + "/health", 10.0):
                tn_fail = 0
            elif not url and not pids_matching(CF_PAT):
                log("no tunnel -> start"); spawn_tunnel(); tn_fail = 0
            else:
                tn_fail += 1
                log(f"public url check failed x{tn_fail} ({url})")
                if tn_fail >= 3:
                    log("restarting tunnel (URL will rotate)")
                    kill_all(pids_matching(CF_PAT)); new = spawn_tunnel(); tn_fail = 0
                    log(f"NEW PUBLIC URL {new}")
        # --- dependencies
        deps, lip = probe_deps()
        down = {f"{n}:{p}" for n, x in deps.items() for p in x["down"]} | {f"lip:{w}" for w, p in lip.items() if p is None}
        for x in sorted(down - prev_down): log("DEP DOWN", x)
        for x in sorted(prev_down - down): log("DEP UP", x)
        prev_down = down
        status = {"time": time.strftime("%F %T"), "ready": healthy and not down, "app_healthy": healthy,
                  "uvicorn_pids": uvp, "tunnel_pids": pids_matching(CF_PAT), "url": read_url(),
                  "tunnel_fail": tn_fail, "deps": deps, "lip": lip, "down": sorted(down)}
        try:
            atomic(f"{ROOT}/supervisor_status.json", json.dumps(status, indent=1))
        except OSError:
            pass
        time.sleep(max(0.5, CYCLE - (time.time() - t)))


if __name__ == "__main__":
    main()
