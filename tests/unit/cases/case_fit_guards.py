"""Fault tests for resource ceilings, ffmpeg timeouts, slot limits and cache validation.

Usage: python case_fit_guards.py <planner.py> <engine.py> [workdir]
"""

import os, sys, json, subprocess, textwrap, time, shutil, tempfile, wave
import numpy as np

PL, PP = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
WD = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else tempfile.mkdtemp(prefix="g066_")
os.makedirs(WD, exist_ok=True)
PY = sys.executable
RES = []


def run(code, env=None, timeout=60, cwd=None):
    e = dict(os.environ)
    e.update(env or {})
    e["PL"], e["PP"] = PL, PP
    pre = (
        "import sys,os,importlib.util,json\n"
        "def load(path,name):\n"
        "    sys.path.insert(0, os.path.dirname(path))\n"
        "    sp=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(sp); sys.modules[name]=m; sp.loader.exec_module(m); return m\n"
    )
    try:
        r = subprocess.run(
            [PY, "-c", pre + textwrap.dedent(code)], env=e, capture_output=True, text=True, timeout=timeout, cwd=cwd
        )
        return r.returncode, r.stdout, r.stderr, False
    except subprocess.TimeoutExpired as ex:
        return -9, (ex.stdout or b"").decode() if isinstance(ex.stdout, bytes) else (ex.stdout or ""), "", True


def check(name, ok, detail=""):
    RES.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + ((" :: " + detail) if detail else ""), flush=True)


def sh(*a, **k):
    return subprocess.run(list(a), capture_output=True, text=True, **k)


# ---- fixtures
def mk_video(path, secs=8, audio=True, faststart=True):
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size=320x240:rate=25:duration={secs}",
    ]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=300:duration={secs}", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if faststart:
        cmd += ["-movflags", "+faststart"]
    cmd += [path]
    assert sh(*cmd).returncode == 0


def dur(path):
    r = sh("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path)
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def decodes(path):
    r = sh("ffmpeg", "-v", "error", "-i", path, "-f", "null", "-")
    return r.returncode == 0 and r.stderr.strip() == ""


def trunc(src, dst, frac):
    b = open(src, "rb").read()
    open(dst, "wb").write(b[: int(len(b) * frac)])


def fresh(name):
    d = os.path.join(WD, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    return d


full_fs = os.path.join(WD, "full_fs.mp4")
mk_video(full_fs, 8, True, True)
full_e = os.path.join(WD, "full_end.mp4")
mk_video(full_e, 8, True, False)
noaud = os.path.join(WD, "noaudio.mp4")
mk_video(noaud, 6, False)

# ---- P1 partial raw.mp4 reused on retry (planner.py:101)
for tag, src in (("faststart", full_fs), ("moov-at-end", full_e)):
    d = fresh("p1_" + tag)
    trunc(src, os.path.join(d, "raw.mp4"), 0.5)
    json.dump({"duration": 8.0, "url": src}, open(os.path.join(d, "meta.json"), "w"))
    rc, o, e, to = run(
        f"""
        m=load(os.environ['PL'],'planner'); m.stage_fetch({src!r}, {d!r}, 0.0, 0.0); print('OK')""",
        timeout=120,
    )
    ok = rc == 0 and abs(dur(os.path.join(d, "raw.mp4")) - 8.0) < 0.5 and decodes(os.path.join(d, "raw.mp4"))
    check(
        f"P1 truncated raw.mp4 ({tag}) is not reused; refetched whole",
        ok,
        f"rc={rc} raw_dur={dur(os.path.join(d, 'raw.mp4')):.2f} {e[-150:]}",
    )

d = fresh("p1_source")
shutil.copy(full_fs, os.path.join(d, "raw.mp4"))
json.dump({"duration": 8.0, "url": full_fs}, open(os.path.join(d, "meta.json"), "w"))
trunc(full_fs, os.path.join(d, "source.mp4"), 0.4)
rc, o, e, to = run(f"m=load(os.environ['PL'],'planner'); m.stage_fetch({full_fs!r}, {d!r}, 0.0, 0.0)", timeout=120)
check(
    "P1 truncated source.mp4 is re-encoded (full decode clean)",
    rc == 0 and decodes(os.path.join(d, "source.mp4")),
    f"rc={rc} decodes={decodes(os.path.join(d, 'source.mp4'))}",
)

d = fresh("p1_atomic")  # crash mid-encode must never leave a file under the final name
rc, o, e, to = run(
    f"""
    m=load(os.environ['PL'],'planner')
    orig=m._run_ffmpeg
    def boom(args, step=0, **k):
        open(args[-1],'wb').write(b'partial-garbage'*100); raise KeyboardInterrupt('simulated kill mid-encode')
    m._run_ffmpeg=boom
    try: m.stage_fetch({full_fs!r}, {d!r}, 0.0, 0.0)
    except BaseException as ex: print('raised', type(ex).__name__)
    print('SRC_EXISTS', os.path.exists(os.path.join({d!r},'source.mp4')))""",
    timeout=60,
)
check(
    "P1 simulated kill mid-encode leaves no source.mp4 under the final name", "SRC_EXISTS False" in o, o.strip()[-100:]
)

# ---- P2 ffmpeg / ffprobe timeouts (planner.py:218)
fake = os.path.join(WD, "fakebin")
os.makedirs(fake, exist_ok=True)
for nm in ("ffmpeg", "ffprobe"):
    open(os.path.join(fake, nm), "w").write("#!/bin/bash\nsleep 40\n")
    os.chmod(os.path.join(fake, nm), 0o755)
penv = {"PATH": fake + ":" + os.environ["PATH"], "PLANNER_FFMPEG_TIMEOUT": "5", "PLANNER_FFPROBE_TIMEOUT": "2"}
t0 = time.time()
rc, o, e, to = run(
    "m=load(os.environ['PL'],'planner')\ntry: m._run_ffmpeg(['-i','x','-y','y.wav'])\nexcept Exception as ex: print('EXC', type(ex).__name__)",
    env=penv,
    timeout=15,
)
check(
    "P2 hung ffmpeg is killed and reported (SubprocessTimeout)",
    (not to) and "EXC SubprocessTimeout" in o,
    f"timeout_hit={to} wall={time.time() - t0:.1f}s out={o.strip()[:60]}",
)
t0 = time.time()
rc, o, e, to = run(
    f"m=load(os.environ['PL'],'planner'); print('DUR', m._ffprobe_duration({full_fs!r}))", env=penv, timeout=12
)
check(
    "P2 hung ffprobe returns (0.0) instead of hanging",
    (not to) and "DUR 0.0" in o,
    f"timeout_hit={to} wall={time.time() - t0:.1f}s",
)

# ---- P3 video without audio (planner.py:268): clean, reported error
d = fresh("p3")
rc, o, e, to = run(
    f"m=load(os.environ['PL'],'planner'); sys.exit(m.main(['--url',{noaud!r},'--out',{d!r}]))", timeout=120
)
pe = os.path.join(d, "planner_error.json")
js = json.load(open(pe)) if os.path.exists(pe) else {}
check(
    "P3 no-audio video -> rc 3, planner_error.json code=no_audio, 'PLANNER_ERROR no_audio' last stderr line",
    rc == 3 and js.get("error") == "no_audio" and e.strip().splitlines()[-1].startswith("PLANNER_ERROR no_audio"),
    f"rc={rc} json={js.get('error')} last={e.strip().splitlines()[-1][:80] if e.strip() else ''}",
)

# ---- P4 memory: full decoded audio in RAM (planner.py:289) + resource ceiling (#12)
big = os.path.join(WD, "big_2h.wav")
if not os.path.exists(big):
    n = 16000 * 7200
    with wave.open(big, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        rng = np.random.default_rng(0)
        for _ in range(7200 // 60):
            w.writeframes((rng.standard_normal(16000 * 60) * 3000).astype("<i2").tobytes())
mem_code = f"""
    import resource
    m=load(os.environ['PL'],'planner')
    r0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    y=m._read_mono16k({big!r})
    seg=y[1000:1000+16000*30]
    n=len(y)
    r1=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print('DELTA_MB', (r1-r0)/1024.0, 'N', n, 'S', float(abs(seg).max()))"""
rc, o, e, to = run(mem_code, timeout=120)
try:
    dmb = float(o.split("DELTA_MB")[1].split()[0])
except Exception:
    dmb = -1
check("P4 reading a 2 h track costs < 100 MB RSS (orig ~700 MB)", 0 <= dmb < 100, f"delta={dmb:.0f} MB {e[-100:]}")
# equivalence vs the wave+float32 reference on a small file
small = os.path.join(WD, "small.wav")
with wave.open(small, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    w.writeframes((np.random.default_rng(1).standard_normal(48000) * 5000).astype("<i2").tobytes())
rc, o, e, to = run(
    f"""
    import wave, numpy as np
    m=load(os.environ['PL'],'planner'); y=m._read_mono16k({small!r})
    w=wave.open({small!r}); ref=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float32)/32768.0
    a=np.asarray(y[0:len(y)]) if not hasattr(y,'to_array') else y.to_array()
    print('EQ', len(y)==len(ref) and bool(np.array_equal(a,ref)) and bool(np.array_equal(np.asarray(y[100:900]),ref[100:900])))""",
    timeout=60,
)
check("P4 memmap reader is sample-identical to the old float32 reader", "EQ True" in o, o.strip()[-60:] + e[-100:])
# resource ceiling: clean error not OOM
d = fresh("p4c")
rc, o, e, to = run(
    f"m=load(os.environ['PL'],'planner'); sys.exit(m.main(['--url',{full_fs!r},'--out',{d!r}]))",
    env={"PLANNER_MAX_DUR_S": "3"},
    timeout=120,
)
js = (
    json.load(open(os.path.join(d, "planner_error.json")))
    if os.path.exists(os.path.join(d, "planner_error.json"))
    else {}
)
check(
    "P4 over-long input -> rc 3 resource_limit with clear message (ceiling)",
    rc == 3 and js.get("error") == "resource_limit",
    f"rc={rc} {js.get('message', '')[:90]}",
)
d = fresh("p4d")
rc, o, e, to = run(
    f"m=load(os.environ['PL'],'planner'); sys.exit(m.main(['--url',{full_fs!r},'--out',{d!r}]))",
    env={"PLANNER_MEM_FACTOR": "1e12"},
    timeout=120,
)
js = (
    json.load(open(os.path.join(d, "planner_error.json")))
    if os.path.exists(os.path.join(d, "planner_error.json"))
    else {}
)
check(
    "P4 RAM guard -> rc 3 resource_limit when working set > 80% of available RAM",
    rc == 3 and js.get("error") == "resource_limit" and "RAM" in js.get("message", ""),
    f"rc={rc} {js.get('message', '')[:90]}",
)

# ---- P5 _gap_points / _make_cuts (planner.py:458) and #12 VAD block offsets  [already fixed: expected to PASS on orig too]
rc, o, e, to = run(
    """
    m=load(os.environ['PL'],'planner')
    g=m._gap_points([[10,20],[30,40]],50)
    ok1 = g==[(0.0,10.0),(20.0,30.0),(40.0,50.0)]
    import numpy as np
    vad=[[i*9.0, i*9.0+8.0] for i in range(40)]          # 8 s speech, 1 s gaps
    m._Y_CACHE=np.zeros(16000*360,dtype=np.float32)+0.01
    cuts=m._make_cuts(vad,360.0,60.0,16000)
    inside=[c for c in cuts[1:] if any(a<c['start']<b for a,b in vad)]
    print('GAP',ok1,'CUTS_INSIDE',len(inside))""",
    timeout=60,
)
check(
    "P5 _gap_points returns silence only; no chunk cut inside speech",
    "GAP True CUTS_INSIDE 0" in o,
    o.strip()[-80:] + e[-150:],
)
rc, o, e, to = run(
    """
    m=load(os.environ['PL'],'planner')
    import numpy as np
    sr=16000; y=np.zeros(sr*1250,dtype=np.float32)
    sp=[(5,9),(598,602),(700,703),(1199,1201),(1240,1245)]
    for a,b in sp: y[a*sr:b*sr]=1
    def ts(seg): 
        v=(seg>0.5).astype(np.int8); d=np.diff(np.concatenate([[0],v,[0]])); st=np.where(d==1)[0]; en=np.where(d==-1)[0]
        return [{'start':a/sr,'end':b/sr} for a,b in zip(st,en)]
    got=m._merge_intervals(m._blocked_vad(y,1250.0,ts))
    ok=len(got)==len(sp) and all(abs(g[0]-a)<0.02 and abs(g[1]-b)<0.02 for g,(a,b) in zip(got,sp))
    print('VADOFF',ok,got)""",
    timeout=60,
)
check(
    "P5/#12 chunked VAD offsets correct on a 1250 s signal",
    "VADOFF True" in o,
    o.strip()[-120:] + e[-150:],
)

# ---- P6 silent ECAPA fallback must be marked degraded (planner.py:568)
d = fresh("p6")
shutil.copy(small, os.path.join(d, "source16k.wav"))
json.dump([[0.1, 2.5]], open(os.path.join(d, "vad.json"), "w"))
rc, o, e, to = run(
    f"""
    sys.modules['pyannote']=None; sys.modules['pyannote.audio']=None      # pyannote import fails -> fallback
    m=load(os.environ['PL'],'planner')
    m._diarize_ecapa=lambda y,sr,vad,od: ('ecapa-agglomerative',[(0.1,2.5,0)])
    b,t=m.stage_diarize({d!r})
    bi=json.load(open(os.path.join({d!r},'spkbackend.json')))
    m.stage_write({d!r}, b, t, {{0:{{'mean_p':0.9}}}}, **({{'degraded':bi.get('degraded'),'degraded_reason':bi.get('degraded_reason')}} if 'degraded' in bi else {{}}))
    sg=json.load(open(os.path.join({d!r},'speakers_global.json')))
    print('DEG', bi.get('degraded'), sg.get('degraded'), (sg.get('degraded_reason') or '')[:40])""",
    timeout=60,
)
check(
    "P6 pyannote failure -> spkbackend.json and speakers_global.json carry degraded=true + reason",
    "DEG True True pyannote failed" in o,
    o.strip()[-100:] + e[-200:],
)

# ---- P7 GENDER_HI/LO validation (planner.py:735)
code7 = """
    m=load(os.environ['PL'],'planner')
    try: r=(m._assign_gender(0.9,30,3), m._assign_gender(0.1,30,3), m._assign_gender(0.5,30,3))
    except Exception as ex: r=('EXC',type(ex).__name__)
    print('G7', r)"""
for env, want, tag in (
    ({"GENDER_HI": "abc"}, "('female', 'male', 'unknown')", "non-numeric HI"),
    ({"GENDER_HI": "nan", "GENDER_LO": "0.4"}, "('female', 'male', 'unknown')", "NaN HI"),
    ({"GENDER_HI": "0.2", "GENDER_LO": "0.8"}, "('female', 'male', 'unknown')", "inverted pair (orig: 0.5 -> female)"),
    ({"GENDER_HI": "5"}, "('female', 'male', 'unknown')", "out-of-range HI (orig: nobody female)"),
    ({"GENDER_HI": "0.7", "GENDER_LO": "0.3"}, "('female', 'male', 'unknown')", "valid custom band"),
):
    rc, o, e, to = run(code7, env=env, timeout=30)
    check(f"P7 GENDER env {tag}", "G7 " + want in o, o.strip()[-70:])

# ---- pipeline items
imp = "os.environ.setdefault('TTS_BASE', os.environ.get('TTS_BASE', '')); P=load(os.environ['PP'],'dub_pipeline')"
# ---- D1 empty TTS_BASE (dub_pipeline.py:39)
rc, o, e, to = run(
    "P=load(os.environ['PP'],'dub_pipeline'); print('TTS', len(P.TTS_BASES), P._next_tts_base())",
    env={"TTS_BASE": ""},
    timeout=120,
)
check(
    "D1 empty TTS_BASE -> default replica list, no modulo-by-zero / late failure",
    "TTS 10 http" in o,
    o.strip()[-100:] + e[-200:],
)
rc, o, e, to = run(
    "P=load(os.environ['PP'],'dub_pipeline'); print('MT', len(P.MT_BASES))", env={"MT_BASES": " , "}, timeout=120
)
check("D1 blank MT_BASES -> defaults (orig: [''] / empty)", "MT 5" in o, o.strip()[-60:] + e[-200:])
# ---- D2 retry policy: total deadline + jitter + no sleep after last failure
rc, o, e, to = run(
    """
    P=load(os.environ['PP'],'dub_pipeline'); import requests, time
    calls=[]
    def fn(timeout): calls.append(time.time()); raise requests.exceptions.ConnectionError('down')
    t0=time.time()
    try: P._with_retries(fn, retries=50, base_delay=1.0)
    except RuntimeError as ex: pass
    el=time.time()-t0
    gaps=[b-a for a,b in zip(calls,calls[1:])]
    print('RETRY', round(el,1), len(calls), [round(g,2) for g in gaps])""",
    env={"DUB_RETRY_DEADLINE": "8"},
    timeout=60,
)
try:
    el = float(o.split("RETRY")[1].split()[0])
except Exception:
    el = 999
check(
    "D2 retry loop bounded by total deadline (8 s) with jittered gaps",
    el <= 9.0,
    o.strip()[-120:] + e[-150:],
)

pre_ = (
    "import sys,os,importlib.util,json\n"
    "def load(path,name):\n"
    "    sys.path.insert(0, os.path.dirname(path))\n"
    "    sp=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(sp); sys.modules[name]=m; sp.loader.exec_module(m); return m\n"
)
# ---- D3 bounded in-flight across processes (#10)
slot = os.path.join(WD, "slots")
shutil.rmtree(slot, ignore_errors=True)
cnt = os.path.join(WD, "cnt.txt")
open(cnt, "w").write("0 0")
worker = f"""
    import fcntl, threading, time
    P=load(os.environ['PP'],'dub_pipeline')
    lim=P._SlotLimiter('tst', 3, slot_dir={slot!r}, wait_s=60)
    def bump(d):
        fd=os.open({cnt!r}, os.O_RDWR); fcntl.flock(fd, fcntl.LOCK_EX)
        cur,mx=map(int, os.read(fd,64).decode().split()); cur+=d; mx=max(mx,cur)
        os.lseek(fd,0,0); os.ftruncate(fd,0); os.write(fd, f'{{cur}} {{mx}}'.encode()); fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)
    def job():
        with lim():
            bump(1); time.sleep(0.15); bump(-1)
    ts=[threading.Thread(target=job) for _ in range(6)]
    [t.start() for t in ts]; [t.join() for t in ts]"""
procs = []
for _ in range(3):
    e_ = dict(os.environ)
    e_["PL"], e_["PP"] = PL, PP
    pre = (
        "import sys,os,importlib.util,json\n"
        "def load(path,name):\n"
        "    sys.path.insert(0, os.path.dirname(path))\n"
        "    sp=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(sp); sys.modules[name]=m; sp.loader.exec_module(m); return m\n"
    )
    procs.append(
        subprocess.Popen(
            [PY, "-c", pre + textwrap.dedent(worker)], env=e_, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    )
errs = [p.communicate(timeout=180)[1] for p in procs]
cur, mx = map(int, open(cnt).read().split())
check(
    "D3 3 procs x 6 threads = 18 callers, limiter(3): global max in flight <= 3 (null: unbounded = 18)",
    mx <= 3 and mx >= 2,
    f"max_inflight={mx} {errs[0][-150:]}",
)
# crash safety + timeout when saturated
hold_py = os.path.join(WD, "hold_slots.py")
open(hold_py, "w").write("""import sys,os,time,importlib.util
sp=importlib.util.spec_from_file_location("dp",os.environ["PP"]); P=importlib.util.module_from_spec(sp); sys.modules["dp"]=P; sys.path.insert(0, os.path.dirname(os.environ["PP"])); sp.loader.exec_module(P)
lim=P._SlotLimiter("kill", 2, slot_dir=os.environ["SD"], wait_s=5)
a=lim.acquire(); b=lim.acquire(); print("HELD", flush=True); time.sleep(300)
""")
rc, o, e, to = run(
    f"""
    import subprocess, time, requests
    P=load(os.environ['PP'],'dub_pipeline')
    env=dict(os.environ, SD={slot!r})
    h=subprocess.Popen([sys.executable,{hold_py!r}],env=env,stdout=subprocess.PIPE,text=True); l=h.stdout.readline(); print('HOLDER',l.strip())
    lim=P._SlotLimiter('kill',2,slot_dir={slot!r},wait_s=1.5)
    try: lim.acquire(); r1='ACQUIRED(bad)'
    except requests.exceptions.Timeout: r1='TIMEOUT'
    h.kill(); h.wait(); time.sleep(0.3)
    tok=lim.acquire(); r2='FREED' if tok else 'NONE'; lim.release(tok)
    print('SAT', r1, r2)""",
    timeout=60,
)
check(
    "D3 saturated limiter times out (requests Timeout -> retried); slots freed when the holder is kill -9'd",
    "SAT TIMEOUT FREED" in o,
    o.strip()[-100:] + e[-300:],
)
rc, o, e, to = run(
    """
    P=load(os.environ['PP'],'dub_pipeline')
    lim=P._SlotLimiter('x',2,slot_dir='/proc/nonexistent/slots',wait_s=1)
    with lim(): print('FAILOPEN')""",
    timeout=30,
)
check("D3 unusable slot dir fails open (service call still proceeds)", "FAILOPEN" in o, o.strip()[-60:] + e[-150:])

# ---- D4 run_ffmpeg timeout in the pipeline
rc, o, e, to = run(
    """
    P=load(os.environ['PP'],'dub_pipeline')
    try: P.run_ffmpeg(['-i','a.wav','b.wav'])
    except Exception as ex: print('EXC', type(ex).__name__)""",
    env={"PATH": fake + ":" + os.environ["PATH"], "PIPE_FFMPEG_TIMEOUT": "3"},
    timeout=15,
)
check(
    "D4 hung ffmpeg in the chunk pipeline raises after PIPE_FFMPEG_TIMEOUT",
    (not to) and "EXC RuntimeError" in o,
    f"timeout_hit={to} {o.strip()[-60:]}",
)

# ---- D5 chunk-audio ceiling in stage_vad (#12)
d = fresh("d5")
shutil.copy(small, os.path.join(d, "src16k.wav"))
rc, o, e, to = run(
    f"""
    from pathlib import Path
    P=load(os.environ['PP'],'dub_pipeline')
    try: P.stage_vad(Path({d!r})); print('RAN')
    except Exception as ex: print('EXC', str(ex)[:80])""",
    env={"PIPE_MAX_AUDIO_S": "1"},
    timeout=180,
)
check(
    "D5 stage_vad refuses audio over PIPE_MAX_AUDIO_S with a clear error",
    "EXC chunk audio is" in o
    or "No module named 'silero_vad'" in o,  # skipped when the optional silero-vad package is absent
    o.strip()[-100:] + e[-200:],
)

# ---- R1 (review round 1 fixes)
# video shorter than the container (audio tail) must stay valid; audio shorter than video must plan
vs = os.path.join(WD, "vshort.mp4")
sh(
    "ffmpeg",
    "-y",
    "-hide_banner",
    "-loglevel",
    "error",
    "-f",
    "lavfi",
    "-i",
    "testsrc=size=160x120:rate=25:duration=6",
    "-f",
    "lavfi",
    "-i",
    "sine=frequency=300:duration=12",
    "-c:v",
    "libx264",
    "-pix_fmt",
    "yuv420p",
    "-c:a",
    "aac",
    "-movflags",
    "+faststart",
    vs,
)
d = fresh("r1a")
shutil.copy(vs, os.path.join(d, "source.mp4"))
rc, o, e, to = run(
    f"m=load(os.environ['PL'],'planner'); print('V', m._check_video({vs!r})); m.stage_audio({d!r}); print('AUDIO OK')",
    timeout=120,
)
check(
    "R1 valid mp4 whose video ends before the audio is accepted (tail decode relative to video end) and its audio extracts",
    "AUDIO OK" in o and "V (True" in o,
    o.strip()[-120:] + e[-200:],
)
ash = os.path.join(WD, "ashort.mp4")
sh(
    "ffmpeg",
    "-y",
    "-hide_banner",
    "-loglevel",
    "error",
    "-f",
    "lavfi",
    "-i",
    "testsrc=size=160x120:rate=25:duration=12",
    "-f",
    "lavfi",
    "-i",
    "sine=frequency=300:duration=5",
    "-c:v",
    "libx264",
    "-pix_fmt",
    "yuv420p",
    "-c:a",
    "aac",
    "-movflags",
    "+faststart",
    ash,
)
d = fresh("r1b")
shutil.copy(ash, os.path.join(d, "source.mp4"))
rc, o, e, to = run(f"m=load(os.environ['PL'],'planner'); m.stage_audio({d!r}); print('AUDIO OK')", timeout=120)
check(
    "R1 valid mp4 whose audio ends 7 s before the video is not rejected by the wav-length check",
    "AUDIO OK" in o,
    o.strip()[-100:] + e[-250:],
)
# cached wav must not bypass the ceiling
d = fresh("r1c")
shutil.copy(full_fs, os.path.join(d, "source.mp4"))
rc, o, e, to = run(f"m=load(os.environ['PL'],'planner'); m.stage_audio({d!r}); print('FIRST OK')", timeout=120)
rc, o, e, to = run(
    f"m=load(os.environ['PL'],'planner')\ntry: m.stage_audio({d!r}); print('CACHED PASSED')\nexcept Exception as ex: print('EXC', type(ex).__name__)",
    env={"PLANNER_MAX_DUR_S": "3"},
    timeout=60,
)
check(
    "R1 duration ceiling also applies when source16k.wav is cached",
    "EXC ResourceLimitError" in o,
    o.strip()[-80:] + e[-150:],
)
# changed start/dur must re-cut source.mp4
d = fresh("r1d")
rc, o, e, to = run(
    f"m=load(os.environ['PL'],'planner'); m.stage_fetch({full_fs!r}, {d!r}, 1.0, 3.0); a=m._ffprobe_duration(os.path.join({d!r},'source.mp4')); m.stage_fetch({full_fs!r}, {d!r}, 2.0, 5.0); b=m._ffprobe_duration(os.path.join({d!r},'source.mp4')); print('DURS', round(a), round(b))",
    timeout=120,
)
check("R1 changed start/dur re-cuts source.mp4 (3.0 s then 5.0 s)", "DURS 3 5" in o, o.strip()[-80:] + e[-200:])
# sweep leaves fresh temp files alone, removes stale ones
d = fresh("r1e")
fr = os.path.join(d, ".g066tmp_fresh.mp4")
st = os.path.join(d, ".g066tmp_stale.mp4")
open(fr, "w").write("x")
open(st, "w").write("x")
os.utime(st, (time.time() - 7200, time.time() - 7200))
rc, o, e, to = run(
    f"m=load(os.environ['PL'],'planner'); m._sweep_tmp({d!r}); print('SW', os.path.exists({fr!r}), os.path.exists({st!r}))",
    timeout=30,
)
check("R1 temp sweep keeps a live planner's fresh temp, removes the stale one", "SW True False" in o, o.strip()[-40:])
# slot limiter: unopenable lock file must fail open, not time out
rc, o, e, to = run(
    """
    import time
    P=load(os.environ['PP'],'dub_pipeline')
    real=os.open
    def deny(path,*a,**k):
        if str(path).endswith('.lock'): raise PermissionError(13,'denied')
        return real(path,*a,**k)
    os.open=deny
    lim=P._SlotLimiter('perm',2,slot_dir=os.environ['SD'],wait_s=20)
    t0=time.time()
    with lim(): print('PROCEEDED', round(time.time()-t0,1))""",
    env={"SD": slot},
    timeout=60,
)
check(
    "R1 EACCES opening the lock file fails open immediately (was: waits out the timeout)",
    "PROCEEDED 0." in o,
    o.strip()[-60:] + e[-200:],
)

# refetching a different input must purge source/wav/derived of the previous one (review r2 BLOCKER)
d = fresh("r2a")
rc, o, e, to = run(
    f"""
    m=load(os.environ['PL'],'planner')
    m.stage_fetch({full_fs!r}, {d!r}, 0.0, 0.0); m.stage_audio({d!r}); json.dump([[0,1]], open(os.path.join({d!r},'vad.json'),'w'))
    m.stage_fetch({noaud!r}, {d!r}, 0.0, 0.0)
    print('R2', round(m._ffprobe_duration(os.path.join({d!r},'source.mp4'))), os.path.exists(os.path.join({d!r},'source16k.wav')), os.path.exists(os.path.join({d!r},'vad.json')))""",
    timeout=180,
)
check(
    "R2 a different input in the same dir replaces source.mp4 (6 s, not the old 8 s) and purges wav + vad",
    "R2 6 False False" in o,
    o.strip()[-60:] + e[-200:],
)

# Qwen r2: cover-art (attached_pic) first video stream must not break the tail decode; over-long source must be re-cut
png = os.path.join(WD, "cover.png")
sh(
    "ffmpeg",
    "-y",
    "-hide_banner",
    "-loglevel",
    "error",
    "-f",
    "lavfi",
    "-i",
    "color=c=red:s=64x64:d=1",
    "-frames:v",
    "1",
    png,
)
ca = os.path.join(WD, "cover.mp4")
r = sh(
    "ffmpeg",
    "-y",
    "-hide_banner",
    "-loglevel",
    "error",
    "-i",
    png,
    "-f",
    "lavfi",
    "-i",
    "testsrc=size=160x120:rate=25:duration=6",
    "-f",
    "lavfi",
    "-i",
    "sine=duration=6",
    "-map",
    "0:v",
    "-map",
    "1:v",
    "-map",
    "2:a",
    "-c:v:0",
    "mjpeg",
    "-pix_fmt:v:0",
    "yuvj420p",
    "-c:v:1",
    "libx264",
    "-pix_fmt:v:1",
    "yuv420p",
    "-c:a",
    "aac",
    "-disposition:v:0",
    "attached_pic",
    "-movflags",
    "+faststart",
    ca,
)
rc, o, e, to = run(f"m=load(os.environ['PL'],'planner'); print('CA', m._check_video({ca!r}))", timeout=60)
check(
    "R3 mp4 whose first video stream is cover art is validated on the real video stream",
    "CA (True" in o,
    o.strip()[-80:] + e[-150:] + r.stderr[-100:],
)
d = fresh("r3b")
shutil.copy(full_fs, os.path.join(d, "raw.mp4"))
shutil.copy(full_fs, os.path.join(d, "source.mp4"))
json.dump({"duration": 8.0, "url": full_fs, "start": 1.0, "dur": 3.0}, open(os.path.join(d, "meta.json"), "w"))
rc, o, e, to = run(
    f"m=load(os.environ['PL'],'planner'); m.stage_fetch({full_fs!r}, {d!r}, 1.0, 3.0); print('LEN', round(m._ffprobe_duration(os.path.join({d!r},'source.mp4'))))",
    timeout=120,
)
check(
    "R3 a source.mp4 LONGER than the requested cut (meta says cut) is re-cut to 3 s",
    "LEN 3" in o,
    o.strip()[-40:] + e[-150:],
)

# ---- R4 (review round 3 / coordinator)
# raw.mp4 with no meta.json cannot prove its identity -> must not be reused for another input
d = fresh("r4a")
shutil.copy(noaud, os.path.join(d, "raw.mp4"))
rc, o, e, to = run(
    f"m=load(os.environ['PL'],'planner'); m.stage_fetch({full_fs!r}, {d!r}, 0.0, 0.0); print('D', round(m._ffprobe_duration(os.path.join({d!r},'source.mp4'))))",
    timeout=120,
)
check(
    "R4 raw.mp4 without meta.json (identity unprovable) is refetched: source is the 8 s input, not the stale 6 s raw",
    "D 8" in o,
    o.strip()[-40:] + e[-150:],
)
# local input replaced in place at the same path
d = fresh("r4b")
loc = os.path.join(WD, "inplace.mp4")
shutil.copy(full_fs, loc)
code = f"""
    m=load(os.environ['PL'],'planner'); m.stage_fetch({loc!r}, {d!r}, 0.0, 0.0); a=round(m._ffprobe_duration(os.path.join({d!r},'source.mp4')))
    import shutil; shutil.copy({noaud!r}, {loc!r}); m.stage_fetch({loc!r}, {d!r}, 0.0, 0.0); b=round(m._ffprobe_duration(os.path.join({d!r},'source.mp4'))); print('IP', a, b)"""
rc, o, e, to = run(code, timeout=180)
check(
    "R4 local input replaced in place (same path) is re-fetched (8 s then 6 s)",
    "IP 8 6" in o,
    o.strip()[-40:] + e[-150:],
)
# multi-audio: track 0 shorter than track 1 must plan
ma = os.path.join(WD, "multiaudio.mp4")
sh(
    "ffmpeg",
    "-y",
    "-hide_banner",
    "-loglevel",
    "error",
    "-f",
    "lavfi",
    "-i",
    "testsrc=size=160x120:rate=25:duration=12",
    "-f",
    "lavfi",
    "-i",
    "sine=frequency=300:duration=5",
    "-f",
    "lavfi",
    "-i",
    "sine=frequency=500:duration=12",
    "-map",
    "0:v",
    "-map",
    "1:a",
    "-map",
    "2:a",
    "-c:v",
    "libx264",
    "-pix_fmt",
    "yuv420p",
    "-c:a",
    "aac",
    "-movflags",
    "+faststart",
    ma,
)
d = fresh("r4c")
shutil.copy(ma, os.path.join(d, "source.mp4"))
rc, o, e, to = run(f"m=load(os.environ['PL'],'planner'); m.stage_audio({d!r}); print('MA OK')", timeout=120)
check(
    "R4 two audio tracks (0:a:0 shorter than 0:a:1): audio extraction accepted",
    "MA OK" in o,
    o.strip()[-40:] + e[-200:],
)
# duration unavailable / video-duration unknown must not reject a decodable file (probe monkeypatched)
patch = """
    m=load(os.environ['PL'],'planner'); _p=m._probe
    def pr(path):
        r=_p(path)
        if r: r=dict(r, duration=%s, v_dur=0.0, a_dur=%s, a0_dur=%s)
        return r
    m._probe=pr
"""
rc, o, e, to = run(
    patch % ("0.0", "0.0", "0.0")
    + f"    print('ND', m._check_video({full_fs!r}))\n    d={fresh('r4d')!r}\n    import shutil; shutil.copy({full_fs!r}, d+'/source.mp4'); m.stage_audio(d); print('ND AUDIO OK')",
    timeout=120,
)
check(
    "R4 container/stream duration unavailable: file accepted on decodability, audio extracted",
    "ND (True" in o and "ND AUDIO OK" in o,
    o.strip()[-120:] + e[-200:],
)
rc, o, e, to = run(patch % ("12.0", "12.0", "12.0") + f"    print('AT', m._check_video({vs!r}))", timeout=120)
check(
    "R4 video stream duration unknown + audio-only tail (video 6 s, container 12 s): accepted",
    "AT (True" in o,
    o.strip()[-120:] + e[-200:],
)
# limiter: FIFO order and one host-wide config
rc, o, e, to = run(
    f"""
    import threading, time
    P=load(os.environ['PP'],'dub_pipeline')
    lim=P._SlotLimiter('fifo', 1, slot_dir={os.path.join(WD, "slots_fifo")!r}, wait_s=60)
    order=[]; mu=threading.Lock()
    def job(i):
        with lim():
            with mu: order.append(i)
            time.sleep(0.15)
    ts=[]
    for i in range(8):
        t=threading.Thread(target=job,args=(i,)); t.start(); ts.append(t); time.sleep(0.06)
    [t.join() for t in ts]
    print('ORDER', order)""",
    timeout=120,
)
check(
    "R4 limiter is FIFO: 8 waiters on 1 slot acquire in arrival order",
    "ORDER [0, 1, 2, 3, 4, 5, 6, 7]" in o,
    o.strip()[-80:] + e[-200:],
)
shutil.rmtree(os.path.join(WD, "slots_cfg"), ignore_errors=True)
cfgw = f"""
    import threading, time, fcntl
    P=load(os.environ['PP'],'dub_pipeline')
    lim=P._SlotLimiter('cfg', int(os.environ['MYLIM']), slot_dir={os.path.join(WD, "slots_cfg")!r}, wait_s=60)
    def bump(d):
        fd=os.open({cnt!r}, os.O_RDWR); fcntl.flock(fd, fcntl.LOCK_EX)
        cur,mx=map(int, os.read(fd,64).decode().split()); cur+=d; mx=max(mx,cur)
        os.lseek(fd,0,0); os.ftruncate(fd,0); os.write(fd, f'{{cur}} {{mx}}'.encode()); fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)
    def job():
        with lim():
            bump(1); time.sleep(0.6); bump(-1)
    ts=[threading.Thread(target=job) for _ in range(6)]
    [t.start() for t in ts]; [t.join() for t in ts]"""
open(cnt, "w").write("0 0")
ps = []
for lim_ in ("2", "40", "8"):
    e_ = dict(os.environ)
    e_.update({"PL": PL, "PP": PP, "MYLIM": lim_})
    ps.append(
        subprocess.Popen(
            [PY, "-c", pre_ + textwrap.dedent(cfgw)], env=e_, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    )
    if lim_ == "2":  # deterministic first writer: start the others only once limits.json holds the 2
        for _ in range(100):
            if os.path.exists(os.path.join(WD, "slots_cfg", "limits.json")):
                break
            time.sleep(0.1)
errs = [p_.communicate(timeout=180)[1] for p_ in ps]
cur, mx = map(int, open(cnt).read().split())
check(
    "R4 processes started with different limits (2, 40, 8) enforce ONE host cap (first writer's 2): max in flight <= 2",
    1 <= mx <= 2,
    f"max_inflight={mx} {errs[0][-150:]}",
)

# ---- R5 (review round 4)
# legacy/local meta.json without src_sig must not vouch for raw.mp4
d = fresh("r5a")
shutil.copy(noaud, os.path.join(d, "raw.mp4"))
json.dump({"duration": 6.0, "url": os.path.realpath(full_fs)}, open(os.path.join(d, "meta.json"), "w"))
rc, o, e, to = run(
    f"m=load(os.environ['PL'],'planner'); m.stage_fetch({full_fs!r}, {d!r}, 0.0, 0.0); print('D', round(m._ffprobe_duration(os.path.join({d!r},'source.mp4'))))",
    timeout=120,
)
check(
    "R5 local meta.json without src_sig (legacy) does not vouch for raw.mp4: refetched (8 s, not stale 6 s)",
    "D 8" in o,
    o.strip()[-40:] + e[-150:],
)
# unknown duration: no EOF-relative seek may be used (fake ffmpeg that fails on -sseof)
fk2 = os.path.join(WD, "fakebin2")
os.makedirs(fk2, exist_ok=True)
real_ff = sh("which", "ffmpeg").stdout.strip()
open(os.path.join(fk2, "ffmpeg"), "w").write(
    '#!/bin/bash\nfor a in "$@"; do [ "$a" = "-sseof" ] && { echo "Option -sseof failed" >&2; exit 1; }; done\nexec %s "$@"\n'
    % real_ff
)
os.chmod(os.path.join(fk2, "ffmpeg"), 0o755)
rc, o, e, to = run(
    patch % ("0.0", "0.0", "0.0") + f"    print('SS', m._check_video({full_fs!r}))",
    env={"PATH": fk2 + ":" + os.environ["PATH"]},
    timeout=120,
)
check(
    "R5 unknown duration: tail decode (-sseof) is skipped, file accepted", "SS (True" in o, o.strip()[-100:] + e[-150:]
)
# limits.json: later stages must be merged into the existing file
sd = os.path.join(WD, "slots_merge")
shutil.rmtree(sd, ignore_errors=True)
rc, o, e, to = run(
    f"""
    P=load(os.environ['PP'],'dub_pipeline')
    os.makedirs({sd!r},exist_ok=True); a=P._SlotLimiter('asr',5,slot_dir={sd!r}); a._resolve_limit()
    b=P._SlotLimiter('tts',7,slot_dir={sd!r}); b._resolve_limit()
    c=P._SlotLimiter('tts',99,slot_dir={sd!r}); c._resolve_limit()
    print('LIM', a.limit, b.limit, c.limit, json.load(open({sd!r}+'/limits.json')))""",
    timeout=60,
)
check(
    "R5 limits.json keeps one host-wide value PER STAGE (later stage merged; tts 7 wins over 99)",
    "LIM 5 7 7 {'asr': 5, 'tts': 7}" in o,
    o.strip()[-100:] + e[-150:],
)
# resource ceiling measures the selected track only (0:a:0 = 5 s; commentary track and container are 12 s)
d = fresh("r5c")
shutil.copy(ma, os.path.join(d, "source.mp4"))
rc, o, e, to = run(
    f"m=load(os.environ['PL'],'planner'); m.stage_audio({d!r}); print('RES OK')",
    env={"PLANNER_MAX_DUR_S": "8"},
    timeout=120,
)
check(
    "R5 ceiling uses the extracted audio track (5 s) not the 12 s commentary track / container",
    "RES OK" in o,
    o.strip()[-60:] + e[-150:],
)
# malformed PIPE_FFMPEG_TIMEOUT
fk3 = os.path.join(WD, "fakebin3")
os.makedirs(fk3, exist_ok=True)
open(os.path.join(fk3, "ffmpeg"), "w").write("#!/bin/bash\nexit 0\n")
os.chmod(os.path.join(fk3, "ffmpeg"), 0o755)
okc = 0
for v in ("", "abc", "nan", "-5"):
    rc, o, e, to = run(
        "P=load(os.environ['PP'],'dub_pipeline'); P.run_ffmpeg(['-i','a.wav','b.wav']); print('RAN')",
        env={"PATH": fk3 + ":" + os.environ["PATH"], "PIPE_FFMPEG_TIMEOUT": v},
        timeout=60,
    )
    okc += "RAN" in o
check(
    "R5 malformed PIPE_FFMPEG_TIMEOUT ('', abc, nan, -5) falls back to the default (4/4 ffmpeg calls run)",
    okc == 4,
    f"{okc}/4 " + e[-100:],
)

# ---- R6 (review round 5): unknown-duration path
cor = os.path.join(WD, "corrupt_mid.mp4")
b_ = bytearray(open(full_fs, "rb").read())
n_ = len(b_)
b_[int(n_ * 0.15) : int(n_ * 0.7)] = os.urandom(int(n_ * 0.7) - int(n_ * 0.15))
open(cor, "wb").write(bytes(b_))
rc, o, e, to = run(patch % ("0.0", "0.0", "0.0") + f"    print('CR', m._check_video({cor!r}))", timeout=120)
check(
    "R6 unknown duration + undecodable media (corrupted body): rejected by the bounded head decode",
    "CR (False" in o,
    o.strip()[-120:] + e[-150:],
)
rc, o, e, to = run(patch % ("0.0", "0.0", "0.0") + f"    print('GD', m._check_video({full_fs!r}))", timeout=120)
check("R6 unknown duration + good media still accepted", "GD (True" in o, o.strip()[-100:] + e[-150:])
d = fresh("r6c")
shutil.copy(full_fs, os.path.join(d, "source.mp4"))
rc, o, e, to = run(f"m=load(os.environ['PL'],'planner'); m.stage_audio({d!r}); print('FIRST')", timeout=120)
rc, o, e, to = run(
    patch % ("0.0", "0.0", "0.0")
    + f"    try: m.stage_audio({d!r}); print('CACHED PASSED')\n    except Exception as ex: print('EXC', type(ex).__name__)",
    env={"PLANNER_MAX_DUR_S": "3"},
    timeout=120,
)
check(
    "R6 cached wav with unknown selected-track length is checked against its REAL length (8 s > ceiling 3 s)",
    "EXC ResourceLimitError" in o,
    o.strip()[-80:] + e[-150:],
)
d = fresh("r6d")
shutil.copy(ash, os.path.join(d, "source.mp4"))
rc, o, e, to = run(
    patch % ("12.0", "0.0", "0.0") + f"    m.stage_audio({d!r}); print('SA OK')",
    env={"PLANNER_MAX_DUR_S": "8"},
    timeout=120,
)
check(
    "R6 selected track length unknown: container/video duration (12 s) is NOT used (audio is 5 s; no false reject / ceiling)",
    "SA OK" in o,
    o.strip()[-80:] + e[-200:],
)

bad = [n for n, ok, _ in RES if not ok]
print(f"\nSUMMARY {len(RES) - len(bad)}/{len(RES)} pass; failing: {bad}")
json.dump(
    RES,
    open(
        os.path.join(
            WD,
            "results_" + os.path.basename(PL).replace(".", "_") + "_" + os.path.basename(os.path.dirname(PL)) + ".json",
        ),
        "w",
    ),
    indent=1,
)
