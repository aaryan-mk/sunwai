"""One module per pipeline stage, each a thin adapter over the engine's tested stage function."""

from sunwai.pipeline.stages.asr import AsrStage
from sunwai.pipeline.stages.audio import AudioStage
from sunwai.pipeline.stages.fetch import FetchStage
from sunwai.pipeline.stages.fit import FitStage
from sunwai.pipeline.stages.mix import MixStage
from sunwai.pipeline.stages.mt import MtStage
from sunwai.pipeline.stages.numbers import NumbersStage
from sunwai.pipeline.stages.prepare import SharedPrepStage
from sunwai.pipeline.stages.separate import SeparateStage
from sunwai.pipeline.stages.slots import SlotsStage
from sunwai.pipeline.stages.speakers import SpeakersStage
from sunwai.pipeline.stages.tts import TtsStage
from sunwai.pipeline.stages.vad import VadStage

__all__ = [
    "AsrStage",
    "AudioStage",
    "FetchStage",
    "FitStage",
    "MixStage",
    "MtStage",
    "NumbersStage",
    "SeparateStage",
    "SharedPrepStage",
    "SlotsStage",
    "SpeakersStage",
    "TtsStage",
    "VadStage",
]
