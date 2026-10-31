"""Active-speaker gate hooks and ROI paste-back: unit tests with synthetic arrays (no model weights)."""

import importlib.util
import sys, os, unittest

_PKG_DIR = os.path.dirname(importlib.util.find_spec("sunwai.lipsync.engine").origin)


def _package_source():
    out = []
    for name in sorted(os.listdir(_PKG_DIR)):
        if name.endswith(".py"):
            out.append(open(os.path.join(_PKG_DIR, name)).read())
    return "\n".join(out)


import numpy as np


def _old_crop(frames, tr, fa, cs=0.4):  # reference implementation (full-frame np.pad)
    from scipy import signal
    import cv2

    b = tr["bbox"]
    s = signal.medfilt(np.maximum(b[:, 3] - b[:, 1], b[:, 2] - b[:, 0]) / 2, 13)
    y = signal.medfilt((b[:, 1] + b[:, 3]) / 2, 13)
    x = signal.medfilt((b[:, 0] + b[:, 2]) / 2, 13)
    out = []
    for k, fr in enumerate(tr["frame"]):
        bs = s[k]
        bsi = int(bs * (1 + 2 * cs))
        img = np.pad(
            frames[fa + int(fr)][:, :, ::-1], ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110, 110)
        )
        my, mx = y[k] + bsi, x[k] + bsi
        face = img[int(my - bs) : int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs))]
        out.append(cv2.resize(np.ascontiguousarray(face), (224, 224)))
    return np.stack(out)


class T(unittest.TestCase):
    def test_engine_multiface_hooks(self):
        src = _package_source()
        for k in (
            "detect_faces",
            "speaking_tracks",
            "cluster_runs",
            "LIP_MULTIFACE",
            "COVERAGE_WARN",
            "speech_s",
            "offloaded",
        ):
            self.assertIn(k, src, k)

    def test_fast_crops_identical_incl_borders(self):
        from sunwai.lipsync import tracking as A

        rng = np.random.RandomState(0)
        frames = rng.randint(0, 255, (30, 240, 320, 3)).astype(np.uint8)
        k = np.arange(30)
        for bbox0 in (
            [100, 60, 180, 160],
            [0, 0, 60, 90],
            [260, 150, 320, 240],
            [-10, 20, 70, 120],
        ):  # interior + all borders
            bb = np.tile(np.array(bbox0, float), (30, 1)) + rng.randn(30, 4)
            tr = {"frame": k, "bbox": bb}
            fast = A.face_crops(A.FaceTrack(frame=k, bbox=bb), frames, 0)
            self.assertTrue(np.array_equal(fast, _old_crop(frames, tr, 0)), bbox0)

    def test_link_tracks_two_faces(self):
        from sunwai.lipsync import tracking as A

        dets = [[{"bbox": [10, 10, 110, 130], "lm": 1}, {"bbox": [300, 20, 400, 140], "lm": 2}] for _ in range(40)]
        tr = A.link_tracks(dets)
        self.assertEqual(len(tr), 2)
        self.assertEqual([len(t.frame) for t in tr], [40, 40])
        self.assertLess(tr[0].bbox[0][0], tr[1].bbox[0][0])

    def test_two_speakers_both_synced_and_clustered(self):
        from sunwai.lipsync import runs as A

        g1, a1 = A.gate_runs([(0, 100)], lambda a, b: [True] * 100)
        g2, a2 = A.gate_runs([(20, 100)], lambda a, b: [True] * 80)
        cl = A.cluster_runs([A.FaceRun("A", r[0], r[1]) for r in g1] + [A.FaceRun("B", r[0], r[1]) for r in g2])
        self.assertEqual(len(cl), 1)
        self.assertEqual({r.key for r in cl[0].runs}, {"A", "B"})  # old code: only the top face

    def test_clusters_are_disjoint_and_ordered(self):
        from sunwai.lipsync import runs as A

        cl = A.cluster_runs([A.FaceRun("A", 0, 50), A.FaceRun("B", 60, 120), A.FaceRun("C", 100, 150)])
        self.assertEqual([(c[0], c[1]) for c in cl], [(0, 50), (60, 150)])

    def test_listener_face_not_synced_speaker_synced(self):
        from sunwai.lipsync import gates as A

        T_ = 60
        dets = [[{"bbox": [10, 10, 130, 150], "lm": 1}, {"bbox": [300, 10, 420, 150], "lm": 2}] for _ in range(T_)]
        crops_by = {}

        def _crops(tr, fr, fa):
            m = np.random.RandomState(int(tr.bbox[0][0])).randint(0, 255, (len(tr.frame), 224, 224, 3)).astype(np.uint8)
            crops_by[int(tr.bbox[0][0])] = 1
            return m

        class _Scorer:
            def score_track(self, cr, wav, t0):
                if len(crops_by) == 1:
                    return np.full(len(cr), 6.0), np.zeros(len(cr))
                return np.full(len(cr), 1.0), np.zeros(len(cr))

        g = A.ActiveSpeakerGate(_Scorer(), A.GateConfig(conf_th=3.0, off_max=6, static_th=0.05), crop_fn=_crops)
        trs = g.speaking_tracks(None, None, 0, T_, dets)
        self.assertEqual(len(trs), 2)
        self.assertTrue(trs[0].mask.all())
        self.assertFalse(trs[1].mask.any())

    def test_coverage_fields_in_engine_report(self):
        src = _package_source()
        for k in ("gated_skip_s", "coverage", "face_s", "synced_s"):
            self.assertIn(k, src, k)


if __name__ == "__main__":
    unittest.main()
