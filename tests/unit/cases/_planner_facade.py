"""Legacy-name view over the ``sunwai.planning`` modules for the fault-injection case scripts.

The case scripts were written against the single-file planner (private names, patched through module globals). This
facade resolves those names to the split modules and applies assignments to every module that holds the name, so a
patch such as ``m._probe = fake`` reaches every caller.
"""

import importlib
import sys
import types

_MODULES = (
    "fetch",
    "audio",
    "vad",
    "chunks",
    "diarization",
    "gender",
    "speakers",
    "video_check",
    "subprocesses",
    "wavio",
    "planner",
)

_ALIASES = {
    "_run_ffmpeg": "run_ffmpeg",
    "_ffprobe_duration": "ffprobe_duration",
    "_read_mono16k": "read_mono16k",
    "_check_video": "check_video",
    "_gap_points": "gap_points",
    "_merge_intervals": "merge_intervals",
    "_blocked_vad": "blocked_vad",
    "_diarize_ecapa": "diarize_ecapa",
    "_assign_gender": "assign_gender",
    "_probe": "probe",
    "_sweep_tmp": "sweep_tmp",
}


class PlannerFacade(types.SimpleNamespace):
    """Attribute access/assignment by legacy name across the planning modules."""

    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "_mods", [importlib.import_module(f"sunwai.planning.{m}") for m in _MODULES])

    def __getattr__(self, name):
        real = _ALIASES.get(name, name)
        for mod in self._mods:
            if hasattr(mod, real):
                return getattr(mod, real)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        real = _ALIASES.get(name, name)
        hit = False
        for mod in self._mods:
            if hasattr(mod, real):
                setattr(mod, real, value)
                hit = True
        if not hit:
            object.__setattr__(self, name, value)

    def _make_cuts(self, vad, duration, chunk, sr):
        return self.make_cuts(vad, duration, chunk, self.__dict__["_Y_CACHE"], sr)

    def stage_diarize(self, out_dir):
        result = importlib.import_module("sunwai.planning.diarization").stage_diarize(out_dir)
        return result.backend, result.turns


def load_planner():
    """The facade, as ``load(path, 'planner')`` used to return the single-file module."""
    return PlannerFacade()


sys.modules.setdefault("_planner_facade", sys.modules[__name__])
