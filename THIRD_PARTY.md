# Third-party software and models

SunwAI's original code is Apache-2.0. Dependencies and model weights keep their own licences. Source-code licence does **not** imply rights to the corresponding checkpoint: check each weight separately. Entries marked **verify** have not been confirmed against the upstream licence text for the exact pinned version and must be checked before a release.

## Libraries

| Component | Role | Licence | Notes |
|---|---|---|---|
| FastAPI | HTTP API | MIT | |
| Uvicorn | ASGI server | BSD-3-Clause | verify |
| Pydantic / pydantic-settings | Validation, settings | MIT | verify |
| requests / httpx | HTTP clients | Apache-2.0 / BSD-3-Clause | verify |
| NumPy, SciPy | Signal processing | BSD-3-Clause | verify |
| librosa | Audio analysis | ISC | verify |
| SoundFile | WAV I/O | BSD-3-Clause | verify; links libsndfile (LGPL) |
| PyTorch | Deep learning runtime | BSD-3-Clause | |
| torchvision | Vision ops | BSD-3-Clause | verify |
| Hugging Face transformers / diffusers / accelerate | Model loading, diffusion | Apache-2.0 | verify |
| pyannote.audio | Speaker diarization | MIT | Code only; see gated weights below |
| SpeechBrain | ECAPA speaker embeddings | Apache-2.0 | |
| Silero VAD | Voice activity detection | MIT | |
| Demucs | Source separation | MIT | |
| yt-dlp | Video download | Unlicense | Use subject to the source platform's terms |
| IndicTransToolkit | IndicTrans2 pre/post-processing | MIT | verify |
| OpenCV, mediapipe, insightface, face-alignment, kornia | Face detection and tracking | Apache-2.0 / BSD / MIT | verify each; insightface models have separate terms (verify) |
| onnxruntime-gpu | ONNX inference | MIT | verify |
| DeepCache | Diffusion speed-up | Apache-2.0 | verify |
| ffmpeg | Media processing (system binary, not bundled) | LGPL-2.1+ or GPL depending on build | verify for your build |
| Deno | JavaScript runtime for yt-dlp (system binary, not bundled) | MIT | verify |
| Ruff, mypy, pytest, pre-commit | Development only | MIT / MIT / MIT / MIT | verify |

## Models and checkpoints (not redistributed)

| Model | Source | Code licence | Weights licence / access | Notes |
|---|---|---|---|---|
| LatentSync 1.6 | ByteDance | Apache-2.0 | verify (Hugging Face model card) | Cloned and downloaded separately by the user |
| SyncNet (used by the active-speaker gate) | Distributed with LatentSync auxiliary checkpoints | verify | verify | Original SyncNet research code has its own terms |
| pyannote speaker-diarization pipelines | pyannote | MIT | **Gated**: accept terms on Hugging Face, use a token | verify per-pipeline conditions; do not redistribute |
| ECAPA-TDNN speaker embeddings | SpeechBrain | Apache-2.0 | verify (model card) | |
| Silero VAD | snakers4 | MIT | MIT | |
| Demucs checkpoints | Meta | MIT | verify | |
| AudioSet-AST event classifier (optional) | verify | verify | verify | Used for non-speech event preservation |
| IndicTrans2 | AI4Bharat | MIT | verify | Local mirror for Manipuri and Sindhi |
| Bodhan ASR / MT / TTS | Hosted API or self-hosted service | n/a | Service terms apply | Voice assets belong to their providers |

## Hosted services

OpenRouter (and Google Gemini when selected) and the Bodhan API are used under their own terms of service. Keys are supplied by the user.

## Content

Do not commit downloaded videos, cookies, credentials, model weights or real job output. Test fixtures must be small and redistributable.
