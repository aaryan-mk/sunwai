"""active-speaker gate for the lip-sync engine.
Decides, per frame, whether the face that LatentSync would re-animate is actually speaking in the ORIGINAL video.
Evidence: SyncNet v2 (already on disk) audio-visual confidence of the S3FD face track against the ORIGINAL audio,
computed on sliding 2 s windows.  Kill switch: env LIP_ASD=0 (parsed robustly).  Fail-open on any exception.
"""
import os, numpy as np, cv2, torch
from scipy import signal
import python_speech_features

FPS = 25


def enabled():
    return str(os.environ.get("LIP_ASD", "1")).strip().lower() not in ("0", "false", "off", "no", "")


def _pdist(f1, f2, vshift=15):
    w = 2 * vshift + 1
    f2p = torch.nn.functional.pad(f2, (0, 0, vshift, vshift))
    u = f2p.unfold(0, w, 1)[: f1.shape[0]]          # [T,1024,w]
    return (f1[:, :, None] - u).norm(dim=1)         # [T,w]


class ActiveSpeakerGate:
    def __init__(self, device="cuda", syncnet_path="checkpoints/auxiliary/syncnet_v2.model", det=None, conf_th=None, win=50, stride=5):
        from eval.syncnet import SyncNetEval
        self.dev = device
        self.sn = SyncNetEval(device=device)
        self.sn.loadParameters(syncnet_path)
        self.sn.__S__.eval()
        self._det = det                                         # S3FD only built if the legacy tracks() path is used
        self.conf_th = float(conf_th if conf_th is not None else os.environ.get("LIP_ASD_CONF", "2.7"))
        self.off_max = int(os.environ.get("LIP_ASD_OFFMAX", "8"))
        self.win, self.stride = win, stride
        self._mf = None
        self.bs = int(os.environ.get("LIP_ASD_BS", "64"))
        self.static_th = float(os.environ.get("LIP_ASD_STATIC", "0.05"))
        # (user): a still photo / frozen face must NOT get mouth movement -> static faces keep their original frames.
        # LIP_ASD_STATIC_SYNC=1 restores the behaviour (sync static faces).
        self.static_sync = str(os.environ.get("LIP_ASD_STATIC_SYNC", "0")).strip().lower() in ("1", "true", "on", "yes")

    # ---- tracks of S3FD faces over frames [fa,fb)
    @property
    def det(self):
        if self._det is None:
            from eval.syncnet_detect import SyncNetDetector
            self._det = SyncNetDetector(device=self.dev, detect_results_dir=os.path.join(os.environ.get("TMPDIR", "/tmp"), "asd_det"))
        return self._det

    def tracks(self, frames, fa, fb):
        dets = []
        for f in range(fa, fb):
            bb = self.det.s3f_detector.detect_faces(frames[f], conf_th=0.9, scales=[0.25])
            dets.append([{"frame": f - fa, "bbox": b[:-1].tolist(), "conf": b[-1]} for b in bb])
        return self.det.track_face(dets, min_track=8, min_face_size=30)

    def _crops(self, tr, frames, fa, cs=0.4):
        b = tr["bbox"]
        s = signal.medfilt(np.maximum(b[:, 3] - b[:, 1], b[:, 2] - b[:, 0]) / 2, 13)
        y = signal.medfilt((b[:, 1] + b[:, 3]) / 2, 13); x = signal.medfilt((b[:, 0] + b[:, 2]) / 2, 13)
        out = []
        for k, fr in enumerate(tr["frame"]):
            bs = s[k]; bsi = int(bs * (1 + 2 * cs))
            fim = frames[fa + int(fr)]
            H, W = fim.shape[:2]
            my, mx = y[k] + bsi, x[k] + bsi
            # same window as the padded-frame version, but slice the source first and pad only the slice
            y0, y1 = int(my - bs) - bsi, int(my + bs * (1 + 2 * cs)) - bsi
            x0, x1 = int(mx - bs * (1 + cs)) - bsi, int(mx + bs * (1 + cs)) - bsi
            cy0, cy1, cx0, cx1 = max(0, y0), min(H, y1), max(0, x0), min(W, x1)
            face = fim[cy0:cy1, cx0:cx1, ::-1]
            if (cy0, cy1, cx0, cx1) != (y0, y1, x0, x1):
                face = np.pad(face, ((cy0 - y0, y1 - cy1), (cx0 - x0, x1 - cx1), (0, 0)), "constant", constant_values=(110, 110))
            out.append(cv2.resize(np.ascontiguousarray(face), (224, 224)))
        return np.stack(out)

    def motion(self, crops):
        """Per-frame mouth-region inter-frame motion (mean abs gray diff, lower-centre of the 224 crop), smoothed over 25 frames.
        A still photo / frozen frame scores ~0 even though SyncNet cannot score it."""
        g = np.stack([cv2.cvtColor(np.ascontiguousarray(c[120:210, 50:174]), cv2.COLOR_BGR2GRAY) for c in crops]).astype(np.float32)
        d = np.zeros(len(g), np.float32)
        if len(g) > 1:
            d[1:] = np.abs(g[1:] - g[:-1]).mean((1, 2)); d[0] = d[1]
        k = min(25, len(d)) | 1
        return signal.medfilt(d, k) if len(d) >= 3 else d

    @torch.no_grad()
    def score_track(self, crops, wav, t0_frame):
        """crops [T,224,224,3] BGR; wav int16 16k mono of the whole chunk; t0_frame = chunk frame idx of crops[0].
        Returns per-frame (conf, off) arrays length T (nan where undefined)."""
        T = len(crops)
        a0 = int(round(t0_frame / FPS * 16000)); a1 = a0 + int(round(T / FPS * 16000))
        aud = wav[a0:a1]
        nan = np.full(T, np.nan)
        if T < 8 or len(aud) < 3200:
            return nan, nan.copy()
        key = (id(wav), len(wav), int(np.asarray(wav[::1600], np.int64).sum()))
        if self._mf is None or self._mf[0] != key or self._mf[2] is not wav:      # strong ref: a recycled id() can never hit a stale cache              # one MFCC pass per chunk, sliced per track (4 mfcc frames / video frame)
            self._mf = (key, np.ascontiguousarray(python_speech_features.mfcc(wav, 16000).T.astype(np.float32)), wav)
        mfcc = self._mf[1][:, 4 * t0_frame: 4 * t0_frame + 4 * T]
        cct = torch.from_numpy(mfcc[None, None])
        im = torch.from_numpy(np.ascontiguousarray(crops)).to(self.dev)                # uint8 [T,224,224,3] on GPU
        n = min(T, int(len(aud) // 640)) - 5
        if n < 3:
            return nan, nan.copy()
        S = self.sn.__S__
        fi, fa_ = [], []
        for i in range(0, n, self.bs):
            idx = range(i, min(n, i + self.bs))
            win = torch.stack([im[v:v + 5] for v in idx]).permute(0, 4, 1, 2, 3).float()      # [B,3,5,224,224]
            fi.append(S.forward_lip(win).cpu())
            fa_.append(S.forward_aud(torch.cat([cct[:, :, :, v * 4:v * 4 + 20] for v in idx], 0).to(self.dev)).cpu())
        D = _pdist(torch.cat(fi), torch.cat(fa_))                                    # [n,31]
        conf = np.full(T, np.nan); off = np.full(T, np.nan); cnt = np.zeros(T)
        acc = np.zeros(T); oacc = np.zeros(T)
        w = min(self.win, n)
        starts = list(range(0, max(1, n - w + 1), self.stride))
        if starts[-1] + w < n:
            starts.append(n - w)
        for a in starts:
            md = D[a:a + w].mean(0)
            c = (md.median() - md.min()).item(); o = 15 - int(md.argmin())
            acc[a:a + w + 4] += c; oacc[a:a + w + 4] += o; cnt[a:a + w + 4] += 1
        m = cnt > 0
        conf[m] = acc[m] / cnt[m]; off[m] = oacc[m] / cnt[m]
        # r2: the last <=4 frames of a track have no full 5-frame lip window -> forward-fill the trailing undefined
        # values so the tail inherits the last decision (otherwise gate_runs() sees a spurious boundary and cross-fades 50%).
        last = np.where(m)[0]
        if len(last) and last[-1] < T - 1:
            conf[last[-1] + 1:] = conf[last[-1]]; off[last[-1] + 1:] = off[last[-1]]
        return conf, off

    def speaking_tracks(self, frames, wav, fa, fb, dets):
        """multi-face gate. dets = engine detections for frames fa..fb-1 (see link_tracks). Every face track is scored
        independently against the ORIGINAL audio; returns [{'tid','frame'(abs),'lm'(abs frame->lm3),'mask'(bool per track frame),'conf'}]."""
        out = []
        for tid, tr in enumerate(link_tracks(dets)):
            cr = self._crops(tr, frames, fa)
            conf, off = self.score_track(cr, wav, fa + int(tr["frame"][0]))
            mot = self.motion(cr)
            ok = (~np.isnan(conf)) & (conf >= self.conf_th) & (np.abs(np.nan_to_num(off, nan=99)) <= self.off_max)
            mask = np.where(mot < self.static_th, bool(getattr(self, "static_sync", False)), ok)
            out.append({"tid": tid, "frame": tr["frame"] + fa, "lm": {fa + k: v for k, v in tr["lm"].items()}, "mask": mask,
                        "conf": conf, "off": off, "mot": mot, "bbox": tr["bbox"]})
        return out

    def speaking(self, frames, wav, fa, fb, centers):
        """centers: dict frame -> (x,y) of the face LatentSync would edit (nose point) or None.
        Returns (mask bool[fb-fa], info). mask True = face is speaking in the ORIGINAL (keep lip-sync)."""
        trs = self.tracks(frames, fa, fb)
        scored = []
        for tr in trs:
            cr = self._crops(tr, frames, fa)
            conf, off = self.score_track(cr, wav, fa + int(tr["frame"][0]))
            scored.append((tr, conf, off, self.motion(cr)))
        mask = np.zeros(fb - fa, bool); confs = np.full(fb - fa, np.nan)
        for f in range(fa, fb):
            c = centers.get(f)
            if c is None:
                continue
            best = None
            for tr, conf, off, mot in scored:
                k = f - fa - int(tr["frame"][0])
                if k < 0 or k >= len(tr["frame"]):
                    continue
                x1, y1, x2, y2 = tr["bbox"][k]
                if x1 <= c[0] <= x2 and y1 <= c[1] <= y2:
                    d = (c[0] - (x1 + x2) / 2) ** 2 + (c[1] - (y1 + y2) / 2) ** 2
                    if best is None or d < best[0]:
                        best = (d, conf[k], off[k], mot[k])
            if best is not None:
                confs[f - fa] = best[1]
                if best[3] < self.static_th:                 # still photo / frozen mouth: keep original frames; LIP_ASD_STATIC_SYNC=1 -> sync
                    mask[f - fa] = bool(getattr(self, "static_sync", False))
                elif not np.isnan(best[1]):
                    mask[f - fa] = best[1] >= self.conf_th and abs(best[2]) <= self.off_max
        return mask, confs


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def link_tracks(dets, min_track=8, iou_th=0.5, max_gap=25, min_face=30):
    """link the ENGINE's own per-frame face detections into tracks (replaces the second S3FD pass).
    dets[k] (k = frame offset) = list of dict(bbox=[x1,y1,x2,y2], lm=lm3 or None). Returns
    [{'frame': np.arange(k0..k1), 'bbox': interpolated [n,4], 'lm': {k: lm3}}]  (frame = offsets into dets)."""
    active, done = [], []
    for k, fd in enumerate(dets):
        used = set()
        for tr in sorted(active, key=lambda t: -t["last_k"]):
            best, bj = iou_th, None
            for j, d in enumerate(fd):
                if j in used:
                    continue
                v = _iou(d["bbox"], tr["box"][-1])
                if v > best:
                    best, bj = v, j
            if bj is not None and k - tr["last_k"] <= max_gap:
                used.add(bj); tr["ks"].append(k); tr["box"].append(list(map(float, fd[bj]["bbox"]))); tr["last_k"] = k
                tr["lm"][k] = fd[bj].get("lm")
        for j, d in enumerate(fd):
            if j not in used:
                active.append({"ks": [k], "box": [list(map(float, d["bbox"]))], "last_k": k, "lm": {k: d.get("lm")}})
        still = []
        for tr in active:
            (still if k - tr["last_k"] <= max_gap else done).append(tr)
        active = still
    out = []
    for tr in done + active:
        if len(tr["ks"]) <= min_track:
            continue
        ks = np.array(tr["ks"]); bb = np.array(tr["box"]); fi = np.arange(ks[0], ks[-1] + 1)
        bi = np.stack([np.interp(fi, ks, bb[:, j]) for j in range(4)], 1)
        if max(np.mean(bi[:, 2] - bi[:, 0]), np.mean(bi[:, 3] - bi[:, 1])) <= min_face:
            continue
        out.append({"frame": fi, "bbox": bi, "lm": tr["lm"]})
    return sorted(out, key=lambda t: t["frame"][0])


def cluster_runs(sub):
    """Group face runs [(key, fa, fb, ...)] into time-DISJOINT clusters [cfa, cfb, [runs]]: faces speaking at the same time share one."""
    clusters = []
    for r in sorted(sub, key=lambda r: (r[1], r[2])):
        if clusters and r[1] < clusters[-1][1]:
            clusters[-1][1] = max(clusters[-1][1], r[2]); clusters[-1][2].append(r)
        else:
            clusters.append([r[1], r[2], [r]])
    return clusters


def gate_runs(runs_abs, speaking_fn, split_into_runs, fps=FPS, min_run_s=0.6, bridge=10, ramp=5):
    """Pure gating logic. speaking_fn(fa,fb)->bool mask over [fa,fb). Returns (gated_runs, alpha{frame:weight<1}).
    Frames outside gated runs keep original pixels; ramp frames at each boundary get a linear crossfade weight."""
    alpha, gated = {}, []
    for (fa, fb) in runs_abs:
        mask = [bool(x) for x in speaking_fn(fa, fb)]
        for (s_, e_) in split_into_runs(mask, fps, min_run_s, bridge):
            r0, r1 = min(ramp, s_), min(ramp, (fb - fa) - e_)
            for j in range(r0):
                alpha[fa + s_ - r0 + j] = (j + 1) / (r0 + 1)
            for j in range(r1):
                alpha[fa + e_ + j] = 1 - (j + 1) / (r1 + 1)
            gated.append((fa + s_ - r0, fa + e_ + r1))
    return gated, alpha


AsdGate = ActiveSpeakerGate      # original name, kept for callers
