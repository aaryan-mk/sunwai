# Models and weights

SunwAI never commits or auto-downloads model weights. Fetch them explicitly with `scripts/download_models.py` or by hand.
Always check each licence before redistributing a checkpoint; see [THIRD_PARTY.md](../THIRD_PARTY.md).

| Model | Used by | How to obtain | Gated |
|---|---|---|---|
| LatentSync 1.6 (UNet, Whisper tiny audio encoder, SyncNet) | lip sync, active-speaker gate | Clone [bytedance/LatentSync](https://github.com/bytedance/LatentSync) at the pinned commit and download its checkpoints (`scripts/download_models.py latentsync`). Set `SUNWAI_LIPSYNC__LATENTSYNC_DIR`. | no |
| pyannote speaker diarization | planning | Accept the model terms on Hugging Face, then set `SUNWAI_PLANNING__HF_TOKEN`. | yes |
| SpeechBrain ECAPA (speaker and language ID) | planning, voice plan | Downloaded by SpeechBrain on first use into the Hugging Face cache. | no |
| Silero VAD, Demucs | planning, mix | Downloaded by their Python packages on first use. | no |
| AudioSet AST (optional) | non-speech event preservation | Hugging Face `MIT/ast-finetuned-audioset-10-10-0.4593`, in the local cache or at `SUNWAI_MIXING__AST_DIR`. | no |
| IndicTrans2 (optional) | local mirror for Manipuri and Sindhi | AI4Bharat checkpoints; set `SUNWAI_MIRRORS__IT2_MODELS_DIR`. | check licence |

## LatentSync layout

The lip-sync engine runs with the upstream checkout as its working directory (`LATENTSYNC_DIR`), because the upstream code
loads `configs/unet/stage2_512.yaml`, `checkpoints/latentsync_unet.pt`, `checkpoints/whisper/tiny.pt` and
`checkpoints/auxiliary/syncnet_v2.model` relative to it. The prototype was validated against upstream commit
`a229c3948406bc2cf6eaf4873e662e70c6a04746`; use that commit unless you re-validate.

```bash
git clone https://github.com/bytedance/LatentSync.git third_party/LatentSync
git -C third_party/LatentSync checkout a229c3948406bc2cf6eaf4873e662e70c6a04746
python scripts/download_models.py latentsync --dest third_party/LatentSync
export SUNWAI_LIPSYNC__LATENTSYNC_DIR=$PWD/third_party/LatentSync
```

## Hosted services

ASR, MT and TTS use the Bodhan API (`BODHAN_MODE=api`) or compatible self-hosted replicas (`BODHAN_MODE=local`). The LLM
uses OpenRouter (`LLM_PROVIDER=openrouter`) or Gemini (`LLM_PROVIDER=gemini`). No weights are involved.
