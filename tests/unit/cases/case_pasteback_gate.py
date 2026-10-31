"""Active-speaker gate hooks and ROI paste-back: unit tests with synthetic arrays (no model weights)."""

import importlib.util
import sys, os, unittest

_ENGINE_PATH = importlib.util.find_spec("sunwai.lipsync.engine").origin
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


def split_into_runs(mask, fps, min_s=1.0, bridge=3):
    from sunwai.lipsync.runs import split_into_runs as impl

    return impl(mask, fps, min_s, bridge)


class T(unittest.TestCase):
    def test_engine_multiface_hooks(self):
        src = open(_ENGINE_PATH).read()
        for k in (
            "detect_faces",
            "speaking_tracks",
            "cluster_runs",
            "LIP_MULTIFACE",
            "COVERAGE_WARN",
            "speech_s",
            'pk["off"]',
        ):
            self.assertIn(k, src, k)

    def test_fast_crops_identical_incl_borders(self):
        from sunwai.lipsync import gates as A

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
            g = object.__new__(A.AsdGate)
            self.assertTrue(np.array_equal(g._crops(tr, frames, 0), _old_crop(frames, tr, 0)), bbox0)

    def test_link_tracks_two_faces(self):
        from sunwai.lipsync import gates as A

        dets = [[{"bbox": [10, 10, 110, 130], "lm": 1}, {"bbox": [300, 20, 400, 140], "lm": 2}] for _ in range(40)]
        tr = A.link_tracks(dets)
        self.assertEqual(len(tr), 2)
        self.assertEqual([len(t["frame"]) for t in tr], [40, 40])
        self.assertLess(tr[0]["bbox"][0][0], tr[1]["bbox"][0][0])

    def test_two_speakers_both_synced_and_clustered(self):
        from sunwai.lipsync import gates as A

        g1, a1 = A.gate_runs([(0, 100)], lambda a, b: [True] * 100, split_into_runs)
        g2, a2 = A.gate_runs([(20, 100)], lambda a, b: [True] * 80, split_into_runs)
        cl = A.cluster_runs([("A", r[0], r[1]) for r in g1] + [("B", r[0], r[1]) for r in g2])
        self.assertEqual(len(cl), 1)
        self.assertEqual({r[0] for r in cl[0][2]}, {"A", "B"})  # old code: only the top face

    def test_clusters_are_disjoint_and_ordered(self):
        from sunwai.lipsync import gates as A

        cl = A.cluster_runs([("A", 0, 50), ("B", 60, 120), ("C", 100, 150)])
        self.assertEqual([(c[0], c[1]) for c in cl], [(0, 50), (60, 150)])

    def test_listener_face_not_synced_speaker_synced(self):
        from sunwai.lipsync import gates as A

        g = object.__new__(A.AsdGate)
        g.conf_th = 3.0
        g.off_max = 6
        g.static_th = 0.05
        g.win = 50
        g.stride = 5
        T_ = 60
        dets = [[{"bbox": [10, 10, 130, 150], "lm": 1}, {"bbox": [300, 10, 420, 150], "lm": 2}] for _ in range(T_)]
        crops_by = {}

        def _crops(tr, fr, fa):
            m = (
                np.random.RandomState(int(tr["bbox"][0][0]))
                .randint(0, 255, (len(tr["frame"]), 224, 224, 3))
                .astype(np.uint8)
            )
            crops_by[int(tr["bbox"][0][0])] = 1
            return m

        g._crops = _crops
        g.score_track = lambda cr, wav, t0: (
            (np.full(len(cr), 6.0), np.zeros(len(cr)))
            if len(crops_by) == 1
            else (np.full(len(cr), 1.0), np.zeros(len(cr)))
        )
        trs = g.speaking_tracks(None, None, 0, T_, dets)
        self.assertEqual(len(trs), 2)
        self.assertTrue(trs[0]["mask"].all())
        self.assertFalse(trs[1]["mask"].any())

    def test_coverage_fields_in_engine_report(self):
        src = open(_ENGINE_PATH).read()
        for k in ("gated_skip_s", "coverage", "face_s", "synced_s"):
            self.assertIn(k, src, k)


if __name__ == "__main__":
    unittest.main()
