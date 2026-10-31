"""Model-resident lip-sync worker for a job root (``DUB_ROOT``).

usage: python -m sunwai.workers.lipsync --root ./jobs --id w7 [--once] [--cfg '{"steps":20}']
The process changes into ``LATENTSYNC_DIR`` (the upstream LatentSync checkout) so its relative ``configs/`` and
``checkpoints/`` paths resolve.
 * claims a chunk with an O_EXCL lock file (dir/lipsync.lock, heartbeat-touched; stale after 90 s) -> N workers are safe
 * priority: chunk the viewer is watching (<root>/focus.json if the server writes it, else newest
   /media/<job>/<lang>/chunks/<n>.mp4 request in server.log), then the chunks after it, then everything else newest dub first
 * output is only published if the dub was not rewritten (segment edit) while we worked
"""
import os, sys, re, json, time, glob, threading, argparse, traceback, socket

STALE_S = 90
CHUNK_RE = re.compile(r"/media/([0-9a-f]+)/([A-Za-z_]+)/chunks/(\d+)\.mp4")


def _sig(path):
    """identity of a published file: (mtime_ns, inode, size). Atomic os.replace publication changes the inode."""
    st = os.stat(path)
    return (st.st_mtime_ns, st.st_ino, st.st_size)


def log(*a):
    print(time.strftime("%F %T"), *a, flush=True)


def focus(root):
    """(job, lang, idx) or None."""
    fj = os.path.join(root, "focus.json")
    try:
        if time.time() - os.path.getmtime(fj) < 900:
            d = json.load(open(fj))
            return d["job"], d["lang"], int(d["chunk"])
    except Exception:
        pass
    sl = os.path.join(root, "server.log")
    try:
        if time.time() - os.path.getmtime(sl) > 900:
            return None
        with open(sl, "rb") as f:
            f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 300000))
            txt = f.read().decode("utf-8", "ignore")
        m = None
        for m in CHUNK_RE.finditer(txt):
            pass
        if m:
            return m.group(1), m.group(2), int(m.group(3))
    except Exception:
        pass
    return None


_SKIP_LOGGED = set()


def lipsync_disabled(root, job):
    """lipopt: True only when <root>/<job>/job.json has "lipsync": false (JSON false). Missing file/field/any other value -> False."""
    try:
        with open(os.path.join(root, job, "job.json")) as f:
            return json.load(f).get("lipsync") is False
    except Exception:
        return False


def candidates(root):
    out = []
    disabled = {}                 # per-call cache: job.json is read once per job per scan, decisions are consistent within a scan
    for dub in glob.glob(f"{root}/*/*/c*/dub_*.mp4") + glob.glob(f"{root}/*/*/dub_*.mp4"):
        if dub.endswith("_lipsync.mp4") or ".part" in dub or ".new" in dub:
            continue
        d = os.path.dirname(dub); lang = os.path.basename(dub)[4:-4]
        lip = os.path.join(d, f"dub_{lang}_lipsync.mp4")
        try:
            dm = os.path.getmtime(dub)
            if time.time() - dm < 3:
                continue
            if os.path.exists(lip) and os.path.getmtime(lip) >= dm:
                continue
            fl = os.path.join(d, "lipsync.failed")
            if os.path.exists(fl) and os.path.getmtime(fl) >= dm:
                try:
                    nf = int(open(fl).read().split("|")[0])
                except Exception:
                    nf = 99
                if nf >= 3 or time.time() - os.path.getmtime(fl) < 60 * nf:
                    continue
        except OSError:
            continue
        if not all(os.path.exists(os.path.join(d, x)) for x in ("dub.wav", "segments.json", "src.mp4")):
            continue
        parts = d.split("/")
        if re.fullmatch(r"c\d{4}", parts[-1]):
            job, lang_dir, idx = parts[-3], parts[-2], int(parts[-1][1:])
        else:
            job, lang_dir, idx = parts[-2], parts[-1], 0
        if job not in disabled:
            disabled[job] = lipsync_disabled(root, job)
        if disabled[job]:          # user turned lip-sync off for this video: no lock, no render, no files
            if job not in _SKIP_LOGGED:
                _SKIP_LOGGED.add(job); log(f"SKIP lipsync=false job {job}")
            continue
        out.append(dict(dir=d, lang=lang, job=job, idx=idx, dub=dub, dm=dm))
    return out


def rank(c, fo):
    if fo and c["job"] == fo[0]:
        same = 0 if c["lang"] == fo[1] else 1
        ahead = c["idx"] - fo[2]
        return (0, same, 1 if ahead < 0 else 0, abs(ahead), 0.0)
    return (1, 0, 0, 0, -c["dm"])


def claim(d):
    lock = os.path.join(d, "lipsync.lock")
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            tok = f"{socket.gethostname()}-{os.getpid()}-{time.time()}"
            os.write(fd, tok.encode()); os.close(fd)
            return lock, tok
        except FileNotFoundError:        # the chunk dir vanished (job deleted/restored mid-queue) -> never kill the worker
            return None
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock) > STALE_S:
                    gone = f"{lock}.stale.{os.getpid()}.{time.time()}"
                    os.rename(lock, gone)      # atomic: only one taker wins the stale lock
                    try:
                        os.remove(gone)
                    except OSError:
                        pass
                    continue
            except OSError:
                pass
            return None
    return None


def owns(lock, tok):
    try:
        return open(lock).read() == tok
    except OSError:
        return False


class Beat(threading.Thread):
    def __init__(self, lock, tok):
        super().__init__(daemon=True); self.lock = lock; self.tok = tok; self.stop = threading.Event()

    def run(self):
        while not self.stop.wait(15):
            try:
                if owns(self.lock, self.tok):
                    os.utime(self.lock, None)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--id", default="w")
    ap.add_argument("--once", action="store_true"); ap.add_argument("--cfg", default="{}")
    a = ap.parse_args()
    cfg = json.loads(a.cfg)
    latentsync_dir = os.environ.get("LATENTSYNC_DIR", "").strip()
    if latentsync_dir:
        os.chdir(latentsync_dir)
    sys.path.insert(0, os.getcwd())
    from sunwai.lipsync import engine as E
    eng = E.Engine()
    stats = os.path.join(a.root, f"lipsync_stats_{a.id}.jsonl")
    done_n = 0
    while True:
        cands = candidates(a.root)
        fo = focus(a.root)
        cands.sort(key=lambda c: rank(c, fo))
        took = False
        for c in cands:
            cl = claim(c["dir"])
            if not cl:
                continue
            lock, tok = cl
            lip = os.path.join(c["dir"], f"dub_{c['lang']}_lipsync.mp4")
            took = True
            if not os.path.isdir(c["dir"]) or not os.path.exists(c["dub"]):      # job disappeared between listing and claim
                log(f"[{a.id}] SKIP vanished {c['dir']}")
                if owns(lock, tok):          # never delete another worker's lock if the dir was recreated meanwhile
                    try:
                        os.remove(lock)
                    except OSError:
                        pass
                break
            hb = Beat(lock, tok); hb.start()
            t0 = time.time()
            log(f"[{a.id}] START {c['job']}/{c['lang']}/c{c['idx']:04d} focus={fo}")
            try:
                if os.path.exists(lip) and os.path.getmtime(lip) >= os.path.getmtime(c["dub"]):
                    continue    # another worker just finished it (finally releases lock)
                dm0 = _sig(c["dub"])
                newp = lip + f".{os.getpid()}.new.mp4"
                rp = os.path.join(c["dir"], f"lipsync.json")
                rep = eng.process_chunk(c["dir"], c["lang"], out_path=newp, report_path=rp + f".{os.getpid()}", **cfg)
                # TOCTOU: re-check the dub signature (mtime_ns+inode+size) IMMEDIATELY before the rename and again
                # right after it; if the dub was rewritten (segment edit) in between, roll the published lip file back.
                stale = _sig(c["dub"]) != dm0 or not owns(lock, tok)
                if not stale:
                    mine = _sig(newp)
                    os.replace(newp, lip)
                    if _sig(c["dub"]) != dm0 or not owns(lock, tok):
                        try:
                            if _sig(lip)[1:] == mine[1:]:      # roll back only OUR file (same inode/size), never a newer worker's
                                os.remove(lip)
                        except OSError:
                            pass
                        stale = True
                if stale:
                    for _p in (newp, rp + f".{os.getpid()}"):
                        try:
                            os.remove(_p)
                        except FileNotFoundError:
                            pass
                    log(f"[{a.id}] DISCARD stale/lost lock {c['dir']}")
                else:
                    os.replace(rp + f".{os.getpid()}", rp)
                    log(f"[{a.id}] DONE {c['job']}/{c['lang']}/c{c['idx']:04d} {time.time()-t0:.0f}s rtf={rep['rtf']} runs={len(rep['runs'])}")
                    with open(stats, "a") as f:
                        f.write(json.dumps(dict(ts=time.time(), dir=c["dir"], wall=round(time.time() - t0, 1), rtf=rep["rtf"], chunk_s=rep["chunk_s"],
                                                synced_frames=rep["synced_frames"], timing=rep["timing_s"])) + "\n")
                done_n += 1
            except Exception as ex:
                log(f"[{a.id}] FAILED {c['dir']}: {ex}")
                traceback.print_exc()
                fl = os.path.join(c["dir"], "lipsync.failed")
                try:
                    nf = int(open(fl).read().split("|")[0]) + 1 if os.path.exists(fl) and os.path.getmtime(fl) >= os.path.getmtime(c["dub"]) else 1
                except Exception:
                    nf = 1
                try:                      # a vanished job dir must never kill the worker
                    open(fl, "w").write(f"{nf}|{ex}")
                except OSError as ex2:
                    log(f"[{a.id}] cannot write failure marker (job gone?): {ex2}")
            finally:
                try:                          # OOM: free this worker's cached GPU memory after every chunk (ok or failed)
                    getattr(eng, "release_cache", lambda: None)()
                except Exception as ex_:
                    log(f"[{a.id}] release_cache error: {ex_}")
                hb.stop.set(); hb.join(timeout=5)
                if owns(lock, tok):
                    try:
                        os.remove(lock)
                    except OSError:
                        pass
            break
        if not took:
            if a.once:
                break
            time.sleep(3)
        elif a.once:
            break


if __name__ == "__main__":
    main()
