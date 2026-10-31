# SunwAI

Dub a video into Indian languages while you watch it.

SunwAI takes a YouTube link and up to three target Indian languages and produces a dubbed video: a separate voice for each speaker, speech timed to fit the original, the background music kept, and lip sync applied to whoever is actually speaking. You can switch the audio language during playback, add a language mid-watch, and edit individual lines.

<!-- TODO: replace with a real recording. Suggested path: docs/media/demo.gif -->
![SunwAI demo (placeholder)](docs/media/demo.gif)

## Key features

- **Up to 3 target languages per job**, dubbed in progressive chunks so playback can start before the whole video is done.
- **Per-speaker voices.** Diarization finds the speakers; a voice plan gives each one a distinct cross-lingual voice, matched to estimated gender.
- **Timing fitted to the original.** An LLM rewrites each translated line to fit its time slot, guarded by a meaning check (numbers, entities, negation) so the rewrite cannot change what was said.
- **Background music kept.** Source separation isolates speech; the dub is ducked into the original background and overlaps/events are preserved.
- **Lip sync on the active speaker only.** LatentSync 1.6 renders the mouth region; an active-speaker gate (SyncNet) plus static-face and side-profile gates decide where to apply it, and the result is pasted back over a region of interest.
- **Live controls.** Switch audio language while watching, add a language mid-watch, edit a line with expression controls (angrier, calmer, and so on) and undo. Lip sync can be turned on or off.
- **Swappable backends.** `BODHAN_MODE=api|local` for ASR, MT and TTS; `LLM_PROVIDER=openrouter|gemini` for rewriting and chat.

## How it works

```mermaid
flowchart LR
    A[YouTube URL] --> B[yt-dlp download]
    B --> C[VAD]
    C --> D[Diarization<br/>pyannote, ECAPA fallback]
    D --> E[Gender + voice plan]
    E --> F[Chunking]
    F --> G[ASR]
    G --> H[MT<br/>Bodhan / IndicTrans2]
    H --> I[LLM fit-to-slot<br/>+ meaning guard]
    I --> J[Number normalisation]
    J --> K[TTS<br/>sentence split + runaway repair]
    K --> L[Separation + mix]
    L --> M[Lip sync<br/>LatentSync + gates]
    M --> N[Dubbed chunk]
```

1. **Download** with yt-dlp (cookies optional; a Deno JS runtime is needed for some YouTube extraction).
2. **Plan** the video: voice-activity detection, speaker diarization (pyannote, with an ECAPA fallback), gender estimation, a per-speaker voice plan, and speech-aware chunking.
3. **Dub each chunk**: ASR, machine translation (Bodhan; IndicTrans2 for Manipuri and Sindhi), LLM fit-to-slot rewrite with a meaning guard, number normalisation, then TTS with per-sentence splitting and repair of runaway syntheses.
4. **Mix**: separate speech from background, align pauses, duck, and limit.
5. **Lip sync** in a persistent GPU worker. Lip-sync readiness is tracked independently of dubbed-audio readiness, so audio plays as soon as it is ready and the lip-synced video replaces it when available.

See [docs/architecture.md](docs/architecture.md) for components, state machines and the stage interface.

## Requirements

- Python 3.10 or newer
- `ffmpeg` and `ffprobe` on `PATH`
- Deno (JavaScript runtime used by yt-dlp for some sites)
- A **Bodhan API key** (or self-hosted Bodhan-compatible ASR/MT/TTS services with `BODHAN_MODE=local`)
- An **OpenRouter API key** (or a Gemini key with `LLM_PROVIDER=gemini`)
- For lip sync: an NVIDIA GPU with roughly 30 GB of memory, CUDA drivers, and the LatentSync weights
- A Hugging Face account and token (pyannote weights are gated)

The API server, the planner and most of the pipeline run on CPU plus remote services. Only lip sync (and optionally local mirrors) need a GPU.

## Quick start

```bash
git clone https://github.com/aaryan-mk/sunwai.git
cd sunwai
uv sync --extra server
cp .env.example .env          # then fill in the keys
uv run sunwai serve --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>, paste a link, choose languages and start a job. For the full pipeline (planning models) and lip sync, install the extras below and fetch the weights.

## Installation

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync --extra server                        # API + web UI
uv sync --extra server --extra pipeline       # + planning/diarization/separation models
uv sync --extra server --extra pipeline --extra lipsync   # + lip sync (GPU)
uv sync --extra dev                           # tests, lint, type checks (CPU only)
```

With pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[server,pipeline,lipsync]"
```

| Extra | Adds | GPU |
|---|---|---|
| *(core)* | Settings, clients, pipeline stages, job logic | no |
| `server` | FastAPI, uvicorn, API and static UI | no |
| `pipeline` | torch, pyannote, speechbrain, silero-vad, demucs, yt-dlp | recommended |
| `lipsync` | Pinned stack for LatentSync, SyncNet gate, face tracking | required |
| `dev` | pytest, ruff, mypy, pre-commit | no |

## Model weights

Weights are never committed to this repository and are not downloaded on import. See `docs/models.md` and `scripts/download_models.py`.

| Model | Where to get it | Notes |
|---|---|---|
| LatentSync 1.6 | Clone [bytedance/LatentSync](https://github.com/bytedance/LatentSync) and download its checkpoints from its Hugging Face model page | Point `SUNWAI_LIPSYNC__LATENTSYNC_DIR` at the checkout. Includes the SyncNet checkpoint used by the active-speaker gate. |
| pyannote speaker diarization | Hugging Face `pyannote/*` pipelines | **Gated.** Accept the model terms with your account, then set `SUNWAI_PLANNING__HF_TOKEN` (or `HF_TOKEN`). |
| ECAPA speaker embeddings | SpeechBrain models on Hugging Face | Fallback diarization and speaker vetting. |
| Silero VAD, Demucs | Fetched by their Python packages | |
| IndicTrans2 (optional) | AI4Bharat on Hugging Face | Only for local mirrors; Manipuri and Sindhi are routed here. |

Check each model's licence before redistributing it; see [THIRD_PARTY.md](THIRD_PARTY.md).

## Configuration

Settings are typed and grouped. Canonical names are `SUNWAI_<GROUP>__<FIELD>`; the original flat names (such as `BODHAN_MODE`) are still accepted. Precedence: explicit values, canonical environment, `.env`, legacy names, built-in defaults. `sunwai config show` prints the effective settings with secrets redacted and `sunwai config map` prints the name mapping. Every variable is listed in [.env.example](.env.example).

| Variable (canonical / legacy) | Purpose |
|---|---|
| `SUNWAI_BODHAN__MODE` / `BODHAN_MODE` | `api` (hosted) or `local` (self-hosted ASR/MT/TTS) |
| `SUNWAI_BODHAN__API_KEY` / `BODHAN_API_KEY` | Bodhan API key |
| `SUNWAI_LLM__PROVIDER` / `LLM_PROVIDER` | `openrouter` or `gemini` |
| `SUNWAI_LLM__OPENROUTER_API_KEY` / `OPENROUTER_API_KEY` | OpenRouter key |
| `SUNWAI_PLANNING__HF_TOKEN` / `HF_TOKEN` | Hugging Face token for gated weights |
| `SUNWAI_PLANNING__YTDLP_COOKIES` / `YTDLP_COOKIES` | Optional cookies file for yt-dlp (keep mode 600) |
| `SUNWAI_SERVER__DUB_ROOT` / `DUB_ROOT` | Directory holding all job state |
| `SUNWAI_SERVER__MAX_TARGETS` / `DUB_MAX_TARGETS` | Maximum target languages per job |
| `SUNWAI_FIT__MEANING_GUARD` / `FIT_MEANING_GUARD` | Reject fit rewrites that change meaning |
| `SUNWAI_TTS__RUNAWAY_LADDER` / `PIPE_RUNAWAY_LADDER` | Enable the runaway-synthesis repair ladder |
| `SUNWAI_LIPSYNC__MODE` / `LIP_MODE` | `proc` to dispatch lip sync, `none` to disable |
| `SUNWAI_LIPSYNC__ASD` / `LIP_ASD` | Active-speaker gate |
| `SUNWAI_LIPSYNC__LATENTSYNC_DIR` / `LATENTSYNC_DIR` | LatentSync checkout |
| `SUNWAI_CONTROL__ACTIONS` / `DUB_ACTIONS` | Enable live controls (actions, chat, expression, add-language) |
| `SUNWAI_CONTROL__TOKEN` / `DUB_CTL_TOKEN` | Control-plane token; control refuses to start without authentication |

A job's own `lipsync=false` is always honoured, and a global lip-sync disable overrides job requests.

## Running

```bash
# API and web UI (single API process; see Limitations)
uv run sunwai serve --host 127.0.0.1 --port 8000

# Persistent lip-sync worker (GPU), pointed at the same jobs root
uv run sunwai worker lipsync --root ./var/jobs --id w1

# Dub one chunk from the command line
uv run sunwai dub --url "<video-url>" --lang hi --out ./var/out --start 0 --dur 90
```

Dubbing work is executed by the API's admitted chunk processes; the lip-sync worker is a separate long-lived process that holds the GPU model and picks up candidates from the jobs root. The web UI is served by the API at `/`.

## Tests

CPU-only, no weights or network required:

```bash
make check                                 # ruff, mypy, pytest
uv run pytest                              # tests/unit and tests/contracts (offline)
uv run pytest tests/gpu -q                 # opt-in, needs GPU + weights (placeholder)
```

## Project structure

```text
sunwai/
├── pyproject.toml            # Dependencies, extras (server, pipeline, lipsync, dev), CLI entry point, tool config
├── uv.lock                   # Locked environment used by CI
├── Makefile                  # install, lint, typecheck, test, check, api, lipsync-worker
├── .env.example              # Documented configuration, no credentials
├── LICENSE  NOTICE  THIRD_PARTY.md
├── src/sunwai/
│   ├── cli.py                # serve, dub, worker, config entry points
│   ├── core/                 # typed settings (SUNWAI_*), legacy env mapping, errors, deadlines, logging, types
│   ├── clients/              # Bodhan endpoints and typed clients, OpenRouter/Gemini LLM client, service calls
│   ├── media/                # ffmpeg helper, WAV I/O, frame ownership
│   ├── language/             # number normalisation, spoken-text selection
│   ├── synthesis/            # sentence splitting, runaway repair ladder, TTS pacing
│   ├── planning/             # fetch, VAD, chunking, diarization, gender, speaker map, voices/ (voice planner)
│   ├── pipeline/             # Stage protocol, runner, DubWorkflow, Segment model; stages/ (one module per stage),
│   │                         #   mixing/, support/, fitting/, shared_prep, voices, resynth, engine (CLI + facade)
│   ├── jobs/                 # add-language scheduler package (focus, ordering, readiness, reconcile, store)
│   ├── control/              # validated actions, chat planner, expression controls, revision store, guards
│   ├── server/               # app factory + container, routers/, services/, file repository, schemas
│   ├── lipsync/              # LatentSync engine split into faces, gates, denoise, paste-back, renderer, report
│   ├── workers/              # lip-sync worker, candidate scanner, claims, supervisor
│   ├── mirrors/              # optional local IndicTrans2 and source-separation services
│   └── data/                 # languages.json, voice_bank.json, voice_rates.json
├── web/                      # static UI: index.html, styles.css, js/{app,effects,cinematic}.js
├── tests/                    # unit/ (incl. cases/ scripts), contracts/ (HTTP routes); integration/ and gpu/ are placeholders
├── docs/                     # architecture, configuration (generated), models
└── scripts/                  # download_models.py, gen_config_docs.py, secret_scan.sh
```

## Limitations and roadmap

This is a first public release of a system that was grown as a prototype. The structure is in place; some parts are still
the original code moved into modules.

**Code quality (honest status)**

- Several long functions remain from the prototype, notably `pipeline/stages/tts.py::stage_tts` (about 490 lines, with
  three inner closures), `stages/mix.py::stage_mix` (about 240), `stages/speakers.py::stage_speakers` and
  `stages/asr.py::stage_asr`. Golden tests for the TTS stage (`tests/unit/test_stage_tts_golden.py`) pin their behaviour so
  they can be split into small classes (`SegmentSynthesizer`, `SentenceSplitter`, `RunawayRepairLadder`, `Mixer`) safely.
- The moved engine modules (stages, `pipeline/support`, `media`, `language`, `synthesis`, `fitting`) are exempt from strict
  lint and type rules in `pyproject.toml` until they are rewritten; new modules are strict. See the `[tool.ruff]` and
  `[tool.mypy]` override blocks for the exact list.
- Segment records are typed (`Segment`) at stage boundaries, but the stage internals still work on the on-disk dicts.

**Service and quality limits**

- The hosted TTS API is rate limited. A job with several languages and many speakers can be throttled; the pipeline retries
  with back-off, falls back to the original audio for lines that still fail, and fails a chunk when more than 20% of its
  speech fell back (`FALLBACK_MAX_FRAC`).
- Songs and music-dominant sections are not dubbed: they are kept as the original audio.
- Lip sync needs a large NVIDIA GPU (about 30 GB) and is skipped for static faces, side profiles and non-speaking faces.
- Jobs live on the filesystem and several guards are in process memory: run one API process.

**Roadmap**

- Split the remaining long functions behind the golden tests and remove the lint/type exemptions module by module.
- Move planning, language and storage logic fully behind the planned `storage/` and typed job repository.
- Contract and integration tests with recorded service replies; CPU-only smoke run in CI.
- More languages, more LLM endpoints, finer expression controls and published quality benchmarks.

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md). Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

## Licence

SunwAI's original code is released under the [Apache License 2.0](LICENSE). Third-party libraries and model weights keep their own licences; see [THIRD_PARTY.md](THIRD_PARTY.md) and [NOTICE](NOTICE). Model weights are not part of this repository.

## Acknowledgements

SunwAI builds on [LatentSync](https://github.com/bytedance/LatentSync), [pyannote.audio](https://github.com/pyannote/pyannote-audio), [SpeechBrain](https://github.com/speechbrain/speechbrain), [Demucs](https://github.com/facebookresearch/demucs), [Silero VAD](https://github.com/snakers4/silero-vad), [yt-dlp](https://github.com/yt-dlp/yt-dlp), [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2), [FastAPI](https://fastapi.tiangolo.com/) and [PyTorch](https://pytorch.org/), and on the Bodhan speech services. Structure borrows patterns from Hugging Face Transformers, WhisperX, nerfstudio, httpx and the FastAPI full-stack template.
