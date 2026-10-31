"""Persistent, batched LatentSync engine for the dubbing studio (replaces per-chunk lipsync_stage.py runs).

Run with cwd = lip/latentsync.  One Engine = models resident on one GPU.
Fixes vs lipsync_stage.py:
  * pipeline() wiped the shared temp dir -> earlier windows lost -> TRUNCATED lipsync mp4 (chunks with >=2 face runs).
    Here no temp dir is shared; frames are assembled in memory and encoded once; duration is verified.
  * models + InsightFace session loaded once (not per chunk / per run).
  * 16-frame windows are batched B-at-a-time through the UNet (windows are independent: same noise, own cond).
  * face landmarks detected once per frame (run detection and affine share it); failed frame inside a run reuses
    the nearest valid landmarks instead of failing the whole run.
"""
import os, sys, json, time, math, subprocess, shutil, tempfile, traceback, gc
import threading, queue
import numpy as np, cv2, torch, kornia
import torch.nn.functional as TF
from einops import rearrange
import torchvision
from torchvision import transforms

from sunwai.lipsync import latentsync as L
try:
    from sunwai.lipsync import gates as ASD     # active-speaker gate (guarded: a missing dep must never stop the worker)
except Exception as _ex:
    ASD = None
    print(f"[engine] asd_gate import failed, gate disabled: {_ex}", flush=True)
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from DeepCache import DeepCacheSDHelper
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature
from latentsync.utils.image_processor import ImageProcessor, load_fixed_mask

FPS = 25
ASD_MIN_RUN_S = 0.6
PASTE_V2 = str(os.environ.get("LIP_PASTE_V2", "1")).strip().lower() not in ("0", "false", "off", "no", "")   # paste-back v2 kill switch
MULTIFACE = str(os.environ.get("LIP_MULTIFACE", "1")).strip().lower() not in ("0", "false", "off", "no", "")
ASD_BRIDGE = 10
F16 = 16
DEFAULTS = dict(steps=8, gs=1.5,   # steps 8 / batch 4 / deepcache 3 = lip_cfg_prod.json
                 deepcache=3, batch=4, pad=0.3, min_run=1.0, bridge=3,
                audio_offset_ms=0, seed=1247, crf=18, max_run_s=20.0,
                pipeline=1)   # pipeline=0 -> original sequential path (kill switch)


def _sh(cmd, **kw):
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw)


def _dur(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path],
                       capture_output=True, text=True)
    return float(r.stdout.strip())


class Engine:
    def __init__(self, device="cuda"):
        t0 = time.time()
        self.dev = torch.device(device)
        self.dt = torch.float16
        cfg = OmegaConf.load("configs/unet/stage2_512.yaml")
        self.cfg = cfg
        sched = DDIMScheduler.from_pretrained("configs")
        aenc = Audio2Feature(model_path="checkpoints/whisper/tiny.pt", device="cuda",
                             num_frames=cfg.data.num_frames, audio_feat_length=cfg.data.audio_feat_length)
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=self.dt)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        unet, _ = UNet3DConditionModel.from_pretrained(OmegaConf.to_container(cfg.model), "checkpoints/latentsync_unet.pt", device="cpu")
        unet = unet.to(dtype=self.dt)
        self.pipe = LipsyncPipeline(vae=vae, audio_encoder=aenc, unet=unet, scheduler=sched).to("cuda")
        self.pipe.unet.eval()
        self.dc = DeepCacheSDHelper(pipe=self.pipe)
        self.dc_interval = None
        mask = load_fixed_mask(cfg.data.resolution, cfg.data.mask_image_path).float().to("cuda")
        self.ip = ImageProcessor(cfg.data.resolution, device="cuda", mask_image=mask)
        self.pipe.image_processor = self.ip
        self.res = cfg.data.resolution
        self.set_deepcache(3)
        try:                      # optional hard cap for workers that share a GPU (env LIP_GPU_FRAC, e.g. 0.5); unset = no cap
            _fr = float(os.environ.get("LIP_GPU_FRAC", "") or 0)
            if 0.05 < _fr < 1.0:
                torch.cuda.set_per_process_memory_fraction(_fr)
        except Exception as ex:
            print(f"[engine] LIP_GPU_FRAC ignored: {ex}", flush=True)
        self.gate = None
        if ASD is not None and ASD.enabled():
            try:
                self.gate = ASD.ActiveSpeakerGate(device="cuda")
            except Exception as ex:      # fail-open: old behaviour
                print(f"[engine] ASD gate unavailable: {ex}", flush=True)
        self.load_s = time.time() - t0
        print(f"[engine] loaded in {self.load_s:.1f}s", flush=True)

    # ---- OOM: give cached GPU blocks back after every chunk (workers share GPUs with planner / sep / TTS)
    def release_cache(self):
        """gc + torch.cuda.empty_cache(); no effect on outputs. Kill switch env LIP_EMPTY_CACHE=0."""
        if str(os.environ.get("LIP_EMPTY_CACHE", "1")).strip().lower() in ("0", "false", "off", "no", ""):
            return
        try:
            gc.collect(); torch.cuda.empty_cache()
        except Exception as ex:
            print(f"[engine] release_cache failed: {ex}", flush=True)

    # ---- DeepCache control
    def set_deepcache(self, interval):
        interval = int(interval)
        if interval == self.dc_interval:
            return
        if self.dc_interval:
            self.dc.disable()
        if interval and interval > 1:
            self.dc.set_params(cache_interval=interval, cache_branch_id=0)
            self.dc.enable()
        self.dc_interval = interval

    # ---- ALL usable faces of a frame (same filters / landmark formula as FaceDetector + landmarks3, but every face, not the biggest)
    def detect_faces(self, frame_rgb, threshold=0.5):
        out = []
        for face in self.ip.face_detector.app.get(frame_rgb):
            x1, y1, x2, y2 = face.bbox.astype(np.int_).tolist()
            w, h = x2 - x1, y2 - y1
            if w < 50 or h < 80 or w / h > 1.5 or w / h < 0.2 or face.det_score < threshold:
                continue
            lmk = np.round(face.landmark_2d_106).astype(np.int_)
            lm3 = np.round([np.mean(lmk[[43, 48, 49, 51, 50]], axis=0), np.mean(lmk[101:106], axis=0), np.mean(lmk[[74, 77, 83, 86]], axis=0)])
            out.append({"bbox": [x1, y1, x2, y2], "lm": lm3})
        return out

    # ---- face landmarks (shared by run detection and affine)
    def landmarks3(self, frame_rgb):
        bbox, lm = self.ip.face_detector(frame_rgb)
        if bbox is None:
            return None
        pt_l = np.mean(lm[[43, 48, 49, 51, 50]], axis=0)
        pt_r = np.mean(lm[101:106], axis=0)
        pt_n = np.mean(lm[[74, 77, 83, 86]], axis=0)
        return np.round([pt_l, pt_r, pt_n])

    def affine_from_lm(self, frame_rgb, lm3):
        face, M = self.ip.restorer.align_warp_face(frame_rgb.copy(), landmarks3=lm3, smooth=True)
        box = [0, 0, face.shape[1], face.shape[0]]
        face = cv2.resize(face, (self.res, self.res), interpolation=cv2.INTER_LANCZOS4)
        return rearrange(torch.from_numpy(face), "h w c -> c h w"), box, M

    # ---- VAE helpers (chunked to bound memory)
    def _vae_encode(self, x, gen):
        outs = []
        for i in range(0, x.shape[0], 32):
            z = self.pipe.vae.encode(x[i:i + 32].to(self.dt)).latent_dist.sample(generator=gen)
            outs.append((z - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor)
        return torch.cat(outs)

    def _vae_decode(self, z):
        outs = []
        z = z / self.pipe.vae.config.scaling_factor + self.pipe.vae.config.shift_factor
        for i in range(0, z.shape[0], 32):
            outs.append(self.pipe.vae.decode(z[i:i + 32]).sample)
        return torch.cat(outs)

    @torch.no_grad()
    def denoise(self, faces_u8, wchunks, steps, gs, batch, seed):
        """faces_u8 [n,3,R,R] uint8 (cpu) ; wchunks list of n whisper tensors -> list of n generated faces [3,R,R] fp16 cuda."""
        dev, dt, pipe = self.dev, self.dt, self.pipe
        n = faces_u8.shape[0]
        nwin = math.ceil(n / F16)
        wins = [(w * F16, min(n, (w + 1) * F16)) for w in range(nwin)]
        full = [w for w, (a, b) in enumerate(wins) if b - a == F16]
        part = [w for w, (a, b) in enumerate(wins) if b - a != F16]
        groups = [full[i:i + batch] for i in range(0, len(full), batch)] + [[w] for w in part]
        gen = torch.Generator(device=dev).manual_seed(seed)
        pipe.scheduler.set_timesteps(steps, device=dev)
        timesteps = pipe.scheduler.timesteps
        cfg_on = gs > 1.0
        res = {}
        lat_hw = self.res // pipe.vae_scale_factor
        noise = torch.randn((1, 4, 1, lat_hw, lat_hw), generator=gen, device=dev, dtype=dt)
        for g in groups:
            idx = [i for w in g for i in range(*wins[w])]
            fw = wins[g[0]][1] - wins[g[0]][0]
            B = len(g)
            faces = faces_u8[idx].to(dev)
            ref_pv, masked_pv, masks = self.ip.prepare_masks_and_masked_images(faces, affine_transform=False)
            mask_l = TF.interpolate(masks, size=(lat_hw, lat_hw)).to(dt)
            mim_l = self._vae_encode(masked_pv, gen)
            ref_l = self._vae_encode(ref_pv, gen)
            to5 = lambda t: rearrange(t, "(b f) c h w -> b c f h w", b=B)
            mask_l, mim_l, ref_l = to5(mask_l), to5(mim_l), to5(ref_l)
            if cfg_on:
                mask_l, mim_l, ref_l = [torch.cat([t] * 2) for t in (mask_l, mim_l, ref_l)]
            aud = torch.stack([wchunks[i] for i in idx]).to(dev, dtype=dt)
            if cfg_on:
                aud = torch.cat([torch.zeros_like(aud), aud])
            lat = (noise.repeat(B, 1, fw, 1, 1) * pipe.scheduler.init_noise_sigma)
            for t in timesteps:
                inp = torch.cat([lat] * 2) if cfg_on else lat
                inp = pipe.scheduler.scale_model_input(inp, t)
                inp = torch.cat([inp, mask_l, mim_l, ref_l], dim=1)
                npred = pipe.unet(inp, t, encoder_hidden_states=aud).sample
                if cfg_on:
                    nu, na = npred.chunk(2)
                    npred = nu + gs * (na - nu)
                lat = pipe.scheduler.step(npred, t, lat, eta=0.0).prev_sample
            dec = self._vae_decode(lat.permute(0, 2, 1, 3, 4).reshape(B * fw, 4, lat_hw, lat_hw))
            dec = LipsyncPipeline.paste_surrounding_pixels_back(dec, ref_pv, 1 - masks, dev, dt)
            for k, i in enumerate(idx):
                res[i] = dec[k]
        return [res[i] for i in range(n)]

    # ---- paste-back is implemented by :class:`sunwai.lipsync.pasteback.PasteBack`
    @property
    def pasteback(self):
        """Lazily built paste-back helper bound to this engine's device and image processor."""
        if getattr(self, "_pasteback", None) is None:
            from sunwai.lipsync.pasteback import PasteBack
            self._pasteback = PasteBack(self.dev, self.ip, PASTE_V2)
        return self._pasteback

    def color_offsets(self, gen_faces, faces_u8):
        """Per-frame seam colour offsets (see ``PasteBack.color_offsets``)."""
        return self.pasteback.color_offsets(gen_faces, faces_u8)

    def restore(self, gen_faces, frames, boxes, mats, coffs=None):
        """Paste generated faces back into ``frames`` (see ``PasteBack.restore``)."""
        return self.pasteback.restore(gen_faces, frames, boxes, mats, coffs)


    # ---- pipelined path (detect/affine/audio  ||  denoise  ||  restore+stream-encode)
    def _run_pipelined(self, frames, out, wins, c, job_dir, dubwav, W, H, N, report, T, tmpo):
        sub_q, res_q = queue.Queue(2), queue.Queue(2)
        alpha = {}
        last_fb = [0]
        asd_rep = {"enabled": self.gate is not None, "frames_before": 0, "frames_after": 0, "runs_after": 0, "secs": 0.0,
                   "speech_s": 0.0, "face_s": 0.0, "synced_s": 0.0, "tracks": 0, "multiface": bool(MULTIFACE)}
        if self.gate is not None:
            asd_rep["conf_th"] = self.gate.conf_th
        report["asd"] = asd_rep
        stop = threading.Event()
        err = []
        tm = {"detect": 0.0, "affine_audio": 0.0, "denoise": 0.0, "restore": 0.0, "enc_write": 0.0}
        tmp = tempfile.mkdtemp(prefix="ls_", dir=os.environ.get("TMPDIR"))
        mx = int(c["max_run_s"] * FPS) if c["max_run_s"] else 0

        def put(q, item):
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.5); return True
                except queue.Full:
                    pass
            return False

        def get(q):
            while True:
                if stop.is_set():
                    return None
                try:
                    return q.get(timeout=0.5)
                except queue.Empty:
                    pass

        def prep():
            try:
                with torch.cuda.stream(torch.cuda.Stream()):
                    lm = {}
                    wav_o = None
                    if self.gate is not None:
                        try:
                            t0 = time.time()
                            _td = tempfile.mkdtemp(prefix="asd_", dir=os.environ.get("TMPDIR"))
                            try:
                                w16 = os.path.join(_td, "o.wav")
                                _sh(["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", os.path.join(job_dir, "src.mp4"), "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", w16])
                                from scipy.io import wavfile
                                wav_o = wavfile.read(w16)[1]
                            finally:
                                shutil.rmtree(_td, ignore_errors=True)
                            asd_rep["secs"] += time.time() - t0
                        except Exception as ex:
                            print(f"[engine] ASD_GATE_ERROR fail-open: {ex}", flush=True); traceback.print_exc()
                            asd_rep["error"] = str(ex)
                    for (a, b) in wins:
                        fa0, fb0 = int(round(a * FPS)), min(N, int(round(b * FPS)))
                        if fb0 <= fa0:
                            continue
                        asd_rep["speech_s"] += (fb0 - fa0) / FPS
                        face_m = np.zeros(fb0 - fa0, bool); sync_m = np.zeros(fb0 - fa0, bool)
                        fruns = None            # [(key, fa, fb, lm_dict)]
                        if self.gate is not None and wav_o is not None and MULTIFACE:
                            try:      # every face track is gated on its own; ALL speaking faces are synced
                                t0 = time.time()
                                dets = [self.detect_faces(frames[f]) for f in range(fa0, fb0)]
                                tm["detect"] += time.time() - t0
                                t0 = time.time()
                                trs = self.gate.speaking_tracks(frames, wav_o, fa0, fb0, dets)
                                torch.cuda.current_stream().synchronize()
                                asd_rep["secs"] += time.time() - t0
                                fruns = []
                                for tr in trs:
                                    tf0 = int(tr["frame"][0]); tfn = len(tr["frame"]); msk = tr["mask"]
                                    face_m[tf0 - fa0: tf0 - fa0 + tfn] = True
                                    asd_rep["frames_before"] += tfn
                                    key = (fa0, tr["tid"])
                                    gated, a_w = ASD.gate_runs([(tf0, tf0 + tfn)], lambda x, y, m=msk: m, L.split_into_runs, FPS, ASD_MIN_RUN_S, ASD_BRIDGE)
                                    for f_, v_ in a_w.items():
                                        alpha[(key, f_)] = v_
                                    for (r0, r1) in gated:
                                        fruns.append((key, r0, r1, tr["lm"])); sync_m[r0 - fa0: r1 - fa0] = True
                                asd_rep["tracks"] += len(trs)
                            except Exception as ex:
                                print(f"[engine] ASD_GATE_ERROR fail-open (multiface): {ex}", flush=True); traceback.print_exc()
                                asd_rep["error"] = str(ex); fruns = None
                                face_m[:] = False; sync_m[:] = False
                        if fruns is None:        # legacy single-face path (kill switch LIP_MULTIFACE=0, no gate, or gate error)
                            t0 = time.time(); mask = []
                            for f in range(fa0, fb0):
                                l3 = self.landmarks3(frames[f]); lm[f] = l3; mask.append(l3 is not None)
                            runs = [(fa0 + s_, fa0 + e_) for (s_, e_) in L.split_into_runs(mask, FPS, c["min_run"], c["bridge"])]
                            tm["detect"] += time.time() - t0
                            asd_rep["frames_before"] += sum(b_ - a_ for a_, b_ in runs)
                            for (a_, b_) in runs:
                                face_m[a_ - fa0: b_ - fa0] = True
                            if self.gate is not None and wav_o is not None and runs:
                                try:
                                    t0 = time.time()
                                    def _spk(fa_, fb_):
                                        centers = {f: lm[f][2] for f in range(fa_, fb_) if lm.get(f) is not None}
                                        return self.gate.speaking(frames, wav_o, fa_, fb_, centers)[0]
                                    gated, a_w = ASD.gate_runs(runs, _spk, L.split_into_runs, FPS, ASD_MIN_RUN_S, ASD_BRIDGE)
                                    for f_, v_ in a_w.items():
                                        alpha[(None, f_)] = v_
                                    runs = gated
                                    torch.cuda.current_stream().synchronize()
                                    asd_rep["secs"] += time.time() - t0
                                except Exception as ex:
                                    print(f"[engine] ASD_GATE_ERROR fail-open: {ex}", flush=True); traceback.print_exc()
                                    asd_rep["error"] = str(ex)
                            for (a_, b_) in runs:
                                sync_m[a_ - fa0: b_ - fa0] = True
                            fruns = [(None, a_, b_, lm) for (a_, b_) in runs]
                        asd_rep["frames_after"] += int(sync_m.sum()); asd_rep["runs_after"] += len(fruns)
                        asd_rep["face_s"] += face_m.sum() / FPS; asd_rep["synced_s"] += sync_m.sum() / FPS
                        sub = []
                        for (key, fa, fb, lmd) in fruns:
                            if mx and fb - fa > mx * 1.3:
                                k = math.ceil((fb - fa) / mx); step = math.ceil((fb - fa) / k)
                                sub += [(key, x, min(fb, x + step), lmd) for x in range(fa, fb, step)]
                            else:
                                sub.append((key, fa, fb, lmd))
                        clusters = ASD.cluster_runs(sub)      # time-disjoint clusters; faces speaking at the same time share one
                        for (cfa, cfb, cruns) in clusters:
                            if stop.is_set():
                                return
                            if cfa < last_fb[0]:      # frames before last_fb may already be in the encoder: clusters must be disjoint + ordered
                                raise RuntimeError(f"pipelined runs overlap ({cfa} < {last_fb[0]})")
                            last_fb[0] = cfb
                            packs = []
                            for (key, fa, fb, lmd) in cruns:
                                ids = list(range(fa, fb))
                                vi = [i for i in ids if lmd.get(i) is not None]
                                if not vi:
                                    continue
                                lms = [lmd[min(vi, key=lambda j: abs(j - i))] if lmd.get(i) is None else lmd[i] for i in ids]
                                t0 = time.time()
                                self.ip.restorer.p_bias = None
                                faces, boxes, mats = [], [], []
                                for k_, i in enumerate(ids):
                                    f_, b_, m_ = self.affine_from_lm(frames[i], lms[k_])
                                    faces.append(f_); boxes.append(b_); mats.append(m_)
                                faces = torch.stack(faces)
                                wav = os.path.join(tmp, f"r_{fa}_{len(packs)}.wav")
                                off = c["audio_offset_ms"] / 1000.0
                                a0 = fa / FPS - off
                                if a0 < 0:
                                    _sh(["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", dubwav, "-af", f"adelay={int(-a0*1000)}:all=1", "-ss", "0", "-t", f"{(fb-fa)/FPS:.3f}", "-ar", "16000", "-ac", "1", wav])
                                else:
                                    _sh(["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-ss", f"{a0:.3f}", "-t", f"{(fb-fa)/FPS:.3f}", "-i", dubwav, "-ar", "16000", "-ac", "1", wav])
                                feat = self.pipe.audio_encoder.audio2feat(wav)
                                wch = self.pipe.audio_encoder.feature2chunks(feature_array=feat, fps=FPS)
                                if len(wch) < len(ids):
                                    if len(ids) - len(wch) > 3:
                                        raise RuntimeError(f"whisper chunks {len(wch)} << frames {len(ids)}")
                                    wch = wch + [wch[-1]] * (len(ids) - len(wch))
                                torch.cuda.current_stream().synchronize()
                                tm["affine_audio"] += time.time() - t0
                                packs.append(dict(key=key, f0=fa, f1=fb, ids=ids, faces=faces, boxes=boxes, mats=mats, wch=wch))
                            if packs and not put(sub_q, (cfa, cfb, packs)):
                                return
            except BaseException as ex:
                err.append(ex); stop.set()
            finally:
                put(sub_q, None)

        enc = {"p": None, "w": 0}

        def restore_worker():
            try:
                with torch.cuda.stream(torch.cuda.Stream()):
                    while True:
                        it = get(res_q)
                        if it is None:
                            break
                        fa, fb, packs = it
                        t0 = time.time()
                        for pk in packs:           # faces speaking at the same time are composited one after the other onto `out`
                            ids, n, key = pk["ids"], pk["n"], pk["key"]
                            gen_ = [g_.to(self.dev) for g_ in pk["gen"]] if pk.get("off") else pk["gen"]
                            rest = self.restore(gen_, [out[i] for i in ids[:n]], pk["boxes"][:n], pk["mats"][:n], coffs=pk.get("coff"))
                            pk["gen"] = None; del gen_
                            for k_ in range(n):
                                a_ = alpha.get((key, ids[k_]), 1.0)
                                if a_ < 1.0:      # crossfade at gate boundaries (as the sequential path)
                                    out[ids[k_]] = np.clip(a_ * np.asarray(rest[k_], np.float32) + (1 - a_) * out[ids[k_]], 0, 255).astype(np.uint8)
                                else:
                                    out[ids[k_]] = rest[k_]
                            del rest      # free this pack's restored frames/GPU faces before the next pack is moved onto the device
                            report["runs"].append({"f0": pk["f0"], "f1": pk["f1"], "t0": round(pk["f0"] / FPS, 2), "t1": round(pk["f1"] / FPS, 2), "n": n,
                                                   "face": None if key is None else f"{key[0]}:{key[1]}"})
                        torch.cuda.current_stream().synchronize()
                        tm["restore"] += time.time() - t0
                        # frames [enc.w, fb) are final now (runs arrive in frame order): stream them to the encoder
                        if enc["p"] is None:
                            cmd = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                                   "-i", dubwav, "-map", "0:v", "-map", "1:a", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", "veryfast",
                                   "-crf", str(c["crf"]), "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", "-f", "mp4", tmpo]
                            enc["p"] = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                        t0 = time.time()
                        if fb > enc["w"]:
                            enc["p"].stdin.write(out[enc["w"]:fb].data); enc["w"] = fb
                        tm["enc_write"] += time.time() - t0
            except BaseException as ex:
                err.append(ex); stop.set()

        th_p = threading.Thread(target=prep, daemon=True); th_r = threading.Thread(target=restore_worker, daemon=True)
        th_p.start(); th_r.start()
        tden = 0.0
        try:
            while True:
                it = get(sub_q)
                if it is None:
                    break
                fa, fb, packs = it
                t0 = time.time()
                for pk in packs:
                    n = len(pk["ids"])
                    pk["gen"] = self.denoise(pk["faces"][:n], pk["wch"][:n], c["steps"], c["gs"], c["batch"], c["seed"])
                    if PASTE_V2:
                        pk["coff"] = self.color_offsets(pk["gen"], pk["faces"][:n])
                    pk["n"] = n; pk.pop("faces"); pk.pop("wch")
                    if len(packs) > 1:      # (Sol2): several faces in one cluster -> never hold more than one pack's faces on the GPU
                        pk["gen"] = [g_.cpu() for g_ in pk["gen"]]; pk["off"] = True
                torch.cuda.current_stream().synchronize()
                tden += time.time() - t0
                if not put(res_q, (fa, fb, packs)):
                    break
        except BaseException as ex:
            err.append(ex); stop.set()
        finally:
            put(res_q, None)
            th_p.join(); th_r.join()
            shutil.rmtree(tmp, ignore_errors=True)
        if err:
            if enc["p"] is not None:
                try:
                    enc["p"].stdin.close()
                except Exception:
                    pass
                enc["p"].kill(); enc["p"].wait()
            try:
                os.remove(tmpo)
            except OSError:
                pass
            raise err[0]
        tm["denoise"] = tden
        t0 = time.time()
        if enc["p"] is not None:
            try:
                if enc["w"] < N:
                    enc["p"].stdin.write(out[enc["w"]:N].data)
                enc["p"].stdin.close()
                if enc["p"].wait() != 0:
                    raise RuntimeError("ffmpeg encode failed")
            except BaseException:
                enc["p"].kill(); enc["p"].wait()
                try:
                    os.remove(tmpo)
                except OSError:
                    pass
                raise
        tm["enc_tail"] = time.time() - t0
        T.update({k: v for k, v in tm.items()})
        asd_rep["secs"] = round(asd_rep["secs"], 2); T["asd"] = asd_rep["secs"]
        report["runs"].sort(key=lambda r: r["f0"])
        return enc["p"] is not None

    # ---- whole chunk
    def process_chunk(self, job_dir, lang, out_path=None, report_path=None, **kw):
        c = dict(DEFAULTS); c.update(kw)
        T = {}
        torch.cuda.reset_peak_memory_stats()
        t_all = time.time()
        src = os.path.join(job_dir, "src.mp4"); dubwav = os.path.join(job_dir, "dub.wav")
        out_path = out_path or os.path.join(job_dir, f"dub_{lang}_lipsync.mp4")
        segments = json.load(open(os.path.join(job_dir, "segments.json")))
        fps0, W, H = L.get_video_info(src)
        t0 = time.time()
        raw = _sh(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", src, "-vf", f"fps={FPS}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]).stdout
        N = len(raw) // (W * H * 3)
        frames = np.frombuffer(raw, np.uint8)[:N * W * H * 3].reshape(N, H, W, 3)
        T["decode"] = time.time() - t0
        total = N / FPS
        self.set_deepcache(c["deepcache"])
        wins = L.create_segments(segments, c["pad"], total)
        if c["pipeline"]:
            out = frames.copy()
            report = {"job_dir": job_dir, "lang": lang, "cfg": c, "runs": [], "frames": N, "windows": len(wins)}
            tmpo = out_path + f".{os.getpid()}.part.mp4"
            had = self._run_pipelined(frames, out, wins, c, job_dir, dubwav, W, H, N, report, T, tmpo)
            if not had:
                shutil.copyfile(os.path.join(job_dir, f"dub_{lang}.mp4"), tmpo)
            return self._finish(job_dir, lang, out_path, report_path, c, T, t_all, tmpo, report, total, N, W, H, out, pre_encoded=True)
        # detection (every frame inside speech windows)
        t0 = time.time()
        lm = {}
        runs_abs = []
        for (a, b) in wins:
            fa, fb = int(round(a * FPS)), min(N, int(round(b * FPS)))
            mask = []
            for f in range(fa, fb):
                l3 = self.landmarks3(frames[f])
                lm[f] = l3
                mask.append(l3 is not None)
            for (s, e) in L.split_into_runs(mask, FPS, c["min_run"], c["bridge"]):
                runs_abs.append((fa + s, fa + e))
        T["detect"] = time.time() - t0
        # only re-animate a face that is SPEAKING in the ORIGINAL video (SyncNet vs original audio); else keep original pixels
        alpha, asd_rep = {}, {"enabled": self.gate is not None, "frames_before": sum(b - a for a, b in runs_abs)}
        if self.gate is not None and runs_abs:
            try:
                t0 = time.time()
                _td = tempfile.mkdtemp(prefix="asd_", dir=os.environ.get("TMPDIR"))
                try:
                    wav16 = os.path.join(_td, "o.wav")
                    _sh(["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", wav16])
                    from scipy.io import wavfile
                    wav = wavfile.read(wav16)[1]
                finally:
                    shutil.rmtree(_td, ignore_errors=True)
                def _spk(fa, fb):
                    centers = {f: lm[f][2] for f in range(fa, fb) if lm.get(f) is not None}
                    return self.gate.speaking(frames, wav, fa, fb, centers)[0]
                gated, alpha = ASD.gate_runs(runs_abs, _spk, L.split_into_runs, FPS, ASD_MIN_RUN_S, ASD_BRIDGE)
                asd_rep.update(frames_after=sum(b - a for a, b in gated), runs_after=len(gated), conf_th=self.gate.conf_th, secs=round(time.time() - t0, 2))
                runs_abs = gated
            except Exception as ex:
                print(f"[engine] ASD_GATE_ERROR fail-open: {ex}", flush=True)
                traceback.print_exc()
                alpha = {}; asd_rep["error"] = str(ex)
        T["asd"] = asd_rep.get("secs", 0.0)
        out = frames.copy()
        report = {"job_dir": job_dir, "lang": lang, "cfg": c, "runs": [], "frames": N, "windows": len(wins), "asd": asd_rep}
        tmp = tempfile.mkdtemp(prefix="ls_", dir=os.environ.get("TMPDIR"))
        tw = th = tt = tr = 0.0
        try:
            sub = []
            mx = int(c["max_run_s"] * FPS) if c["max_run_s"] else 0
            for (fa, fb) in runs_abs:
                if mx and fb - fa > mx * 1.3:
                    k = math.ceil((fb - fa) / mx); step = math.ceil((fb - fa) / k)
                    sub += [(x, min(fb, x + step)) for x in range(fa, fb, step)]
                else:
                    sub.append((fa, fb))
            for (fa, fb) in sub:
                ids = list(range(fa, fb))
                vi = [i for i in ids if lm.get(i) is not None]
                if not vi:
                    continue
                import bisect
                lms = [lm[min(vi, key=lambda j: abs(j - i))] if lm.get(i) is None else lm[i] for i in ids]
                t0 = time.time()
                self.ip.restorer.p_bias = None
                faces, boxes, mats = [], [], []
                for k, i in enumerate(ids):
                    f_, b_, m_ = self.affine_from_lm(frames[i], lms[k])
                    faces.append(f_); boxes.append(b_); mats.append(m_)
                faces = torch.stack(faces)
                # audio for this run
                wav = os.path.join(tmp, f"r_{fa}.wav")
                off = c["audio_offset_ms"] / 1000.0
                a0 = fa / FPS - off     # positive offset = audio features taken from EARLIER audio (video lags audio less)
                if a0 < 0:
                    _sh(["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", dubwav, "-af", f"adelay={int(-a0*1000)}:all=1", "-ss", "0", "-t", f"{(fb-fa)/FPS:.3f}", "-ar", "16000", "-ac", "1", wav])
                else:
                    _sh(["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-ss", f"{a0:.3f}", "-t", f"{(fb-fa)/FPS:.3f}", "-i", dubwav, "-ar", "16000", "-ac", "1", wav])
                feat = self.pipe.audio_encoder.audio2feat(wav)
                wch = self.pipe.audio_encoder.feature2chunks(feature_array=feat, fps=FPS)
                if len(wch) < len(ids):
                    if len(ids) - len(wch) > 3:
                        raise RuntimeError(f"whisper chunks {len(wch)} << frames {len(ids)}")
                    wch = wch + [wch[-1]] * (len(ids) - len(wch))
                n = len(ids)
                tw += time.time() - t0
                t0 = time.time()
                gen_faces = self.denoise(faces[:n], wch[:n], c["steps"], c["gs"], c["batch"], c["seed"])
                torch.cuda.synchronize()
                tt += time.time() - t0
                t0 = time.time()
                rest = self.restore(gen_faces, [frames[i] for i in ids[:n]], boxes[:n], mats[:n],
                                    coffs=(self.color_offsets(gen_faces, faces[:n]) if PASTE_V2 else None))
                for k in range(n):
                    a_ = alpha.get(ids[k], 1.0)
                    if a_ < 1.0:      # crossfade at gate boundaries
                        out[ids[k]] = np.clip(a_ * np.asarray(rest[k], np.float32) + (1 - a_) * frames[ids[k]], 0, 255).astype(np.uint8)
                    else:
                        out[ids[k]] = rest[k]
                tr += time.time() - t0
                report["runs"].append({"f0": fa, "f1": fb, "t0": round(fa / FPS, 2), "t1": round(fb / FPS, 2), "n": n})
                del faces, gen_faces, rest
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        T["affine_audio"] = tw; T["denoise"] = tt; T["restore"] = tr
        tmpo = out_path + f".{os.getpid()}.part.mp4"
        return self._finish(job_dir, lang, out_path, report_path, c, T, t_all, tmpo, report, total, N, W, H, out, pre_encoded=False)

    def _finish(self, job_dir, lang, out_path, report_path, c, T, t_all, tmpo, report, total, N, W, H, out, pre_encoded):
        dubwav = os.path.join(job_dir, "dub.wav")
        # encode once + mux dub audio (atomic)
        t0 = time.time()
        if pre_encoded:
            pass
        elif not report["runs"]:
            shutil.copyfile(os.path.join(job_dir, f"dub_{lang}.mp4"), tmpo)
        else:
            cmd = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                   "-i", dubwav, "-map", "0:v", "-map", "1:a", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", "veryfast",
                   "-crf", str(c["crf"]), "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", "-f", "mp4", tmpo]
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            p.stdin.write(out.data); p.stdin.close()
            if p.wait() != 0:
                raise RuntimeError("ffmpeg encode failed")
        T["encode"] = time.time() - t0
        d_out, d_src = _dur(tmpo), _dur(os.path.join(job_dir, f"dub_{lang}.mp4"))
        report["dur_out"], report["dur_ref"] = round(d_out, 3), round(d_src, 3)
        if abs(d_out - d_src) > 0.25:
            os.remove(tmpo)
            raise RuntimeError(f"duration mismatch out={d_out:.2f} ref={d_src:.2f}")
        os.replace(tmpo, out_path)
        T["total"] = time.time() - t_all
        synced = sum(r["n"] for r in report["runs"])
        report["timing_s"] = {k: round(v, 2) for k, v in T.items()}
        report["synced_frames"] = synced
        report["chunk_s"] = round(total, 2)
        report["rtf"] = round(T["total"] / max(total, 1e-6), 3)
        report["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
        a_ = report.get("asd")
        if isinstance(a_, dict) and "speech_s" in a_:          # item 15: face-run vs speech coverage (report-only, never blocks publication)
            a_["gated_skip_s"] = round(max(0.0, a_["face_s"] - a_["synced_s"]), 2)
            a_["coverage"] = round(a_["synced_s"] / a_["speech_s"], 3) if a_["speech_s"] > 0 else None
            for k_ in ("speech_s", "face_s", "synced_s"):
                a_[k_] = round(a_[k_], 2)
            if a_["face_s"] >= 1.0 and a_["coverage"] is not None and a_["coverage"] < 0.8:
                print(f"[engine] COVERAGE_WARN {job_dir}: synced {a_['synced_s']}s of {a_['speech_s']}s speech ({a_['coverage']:.0%}); "
                      f"face on screen {a_['face_s']}s, gated-skip {a_['gated_skip_s']}s", flush=True)
        rp = report_path or os.path.join(job_dir, "lipsync.json")
        json.dump(report, open(rp + ".tmp", "w"), indent=1); os.replace(rp + ".tmp", rp)
        print(f"[engine] {job_dir} chunk={total:.1f}s runs={len(report['runs'])} synced={synced}f total={T['total']:.1f}s "
              f"(dec {T['decode']:.1f} det {T['detect']:.1f} aff+aud {T['affine_audio']:.1f} den {T['denoise']:.1f} rest {T['restore']:.1f} enc {T['encode']:.1f}) pipeline={c['pipeline']}", flush=True)
        return report
