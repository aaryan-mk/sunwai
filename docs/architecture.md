# Architecture

SunwAI is one Python package (`src/sunwai`), one FastAPI application, and separate worker processes. Job state lives on the filesystem. GPU models are loaded inside workers, never at import time.

Dependency rule: routers, then services, then domain modules, then clients and storage. Domain modules never import FastAPI. There is no plugin framework and no dependency-injection container; `server/app.py` (`create_app`) with `server/container.py` is a small composition root.

## Components

```mermaid
flowchart LR
    Web["Static web UI"] --> API["FastAPI routers"]
    API --> Services["Job, edit, media services"]
    API --> Control["Validated control actions"]
    Control --> Services
    Services --> Store["Job repository + revision store"]
    Services --> Queue["Admission and scheduling"]
    Queue --> Dub["Dub worker"]
    Dub --> Planner["Planner"]
    Dub --> Pipeline["Typed dubbing stages"]
    Pipeline --> Clients["Bodhan / local mirrors / OpenRouter"]
    Dub --> Artifacts["Versioned media artifacts"]
    Artifacts --> Lip["Persistent lip-sync worker"]
    Lip --> Artifacts
    Lip --> Store
    Dub --> Store
    API --> Artifacts
```

| Component | Package | Responsibility |
|---|---|---|
| Web UI | `web/` | Player, transcript editing, language switching, action previews and chat. No framework. |
| API | `server/` | App factory, routers, services, public schemas. Translates domain errors to HTTP. |
| Control plane | `control/` | One validated action path (`ActionService`) used by buttons, transcript edits and chat; append-only revision/event store; auth, rate limits, spend caps. |
| Jobs | `jobs/` | Add-a-language scheduling (`jobs/scheduler/`). Job admission, priority and the file repository live in `server/`. |
| Planning | `planning/` | VAD, diarization, speaker vetting, gender, voice plan, chunk boundaries, plan signature. |
| Pipeline | `pipeline/` | Stage protocol, runner (cache, retry, cancellation, timing), explicit workflow, stages, fitting, synthesis, mixing. |
| Lip sync | `lipsync/` | Face tracking, gates, LatentSync adapter, paste-back, publication checks. |
| Clients | `clients/` | ASR, MT, TTS, LID and LLM protocols with Bodhan, local-mirror and OpenRouter adapters; yt-dlp adapter. |
| Workers | `workers/` | Persistent lip-sync worker (`LipSyncWorker`) and the service supervisor. |

## Data flow

```mermaid
flowchart TD
    URL[URL + languages + options] --> Create[POST /api/jobs]
    Create --> Plan[Planner: download, VAD, diarize, gender, voice plan, chunks]
    Plan --> PlanFile[(plan + speaker map)]
    PlanFile --> Shared[Shared prep per chunk: audio, slots, speakers]
    Shared --> L1[Language A: ASR to mix]
    Shared --> L2[Language B: ASR to mix]
    L1 --> Dubbed[(dubbed chunk media)]
    L2 --> Dubbed
    Dubbed --> Cand[Lip-sync candidates]
    Cand --> Lip[Lip-sync worker]
    Lip --> LipOut[(lip-synced chunk)]
    Dubbed --> Media[GET /media/...]
    LipOut --> Media
    Media --> Player[Player]
```

Per chunk and language, the stage order is:

`fetch → audio → separate → vad → slots → speakers → asr → mt → fit → numbers → tts → mix → output`

Language-independent work (source cut, audio, separation, slots) is shared across target languages. The workflow in `pipeline/workflow.py` is an explicit, typed sequence of calls, not a list of untyped stages. Lip sync runs afterwards in its own worker and is not a stage of the dub workflow.

## State machines

Three levels are tracked separately: job, language track, chunk. Lip-sync readiness is independent of dubbed-media readiness.

### Job

```mermaid
stateDiagram-v2
    [*] --> QUEUED
    QUEUED --> PLANNING
    PLANNING --> RUNNING
    RUNNING --> READY
    PLANNING --> FAILED
    RUNNING --> FAILED
    FAILED --> QUEUED: explicit retry
    READY --> RUNNING: accepted edit / add-language
    QUEUED --> CANCEL_REQUESTED
    PLANNING --> CANCEL_REQUESTED
    RUNNING --> CANCEL_REQUESTED
    CANCEL_REQUESTED --> CANCELLED
```

`READY` means the currently requested work satisfies the readiness policy; partially playable ranges are reported separately. Recovery reclaims expired work through events, not by resetting status fields.

### Chunk (per language)

```mermaid
stateDiagram-v2
    [*] --> PENDING
    PENDING --> CLAIMED: lease granted
    CLAIMED --> RUNNING: stages executing
    RUNNING --> DUBBED: output validated + published
    RUNNING --> FAILED: non-retryable / out of attempts
    CLAIMED --> PENDING: lease expired
    RUNNING --> PENDING: lease expired
    DUBBED --> LIPSYNC_PENDING: lip sync requested
    LIPSYNC_PENDING --> LIPSYNCED: candidate published
    LIPSYNC_PENDING --> DUBBED: lip sync skipped or failed (audio stays valid)
    DUBBED --> PENDING: edit invalidates downstream
    FAILED --> PENDING: retry
```

The chunk diagram is a conceptual summary; the authoritative states are in `jobs/state.py`.

## Stage interface

```python
class Stage(Protocol[InT, OutT]):
    name: str
    version: str
    policy: StagePolicy  # max_attempts, retry_backoff_s, timeout_s, cacheable

    def cache_key(self, value: InT, ctx: ChunkContext) -> str: ...
    def run(self, value: InT, ctx: ChunkContext) -> OutT: ...
    def validate(self, value: OutT, ctx: ChunkContext) -> None: ...
```

`ChunkContext` is immutable (job, language, chunk index, plan signature, generation, revision, work directory, settings, deadline, cancellation token). Clients are injected into stage constructors; the context is not a service locator.

The runner (`pipeline/runner.py`):

- builds cache keys from input digests, stage and config versions, model versions, language, voice/style and plan/revision identity;
- validates cached artifacts rather than trusting file existence;
- writes into attempt-local paths and publishes artifacts and manifest atomically under ownership checks;
- retries only classified transient failures, within one deadline (HTTP retries belong to the clients);
- records cache hits, queue wait, attempts, execution time and degraded outcomes separately;
- checks cancellation before expensive work and before publication, so a superseded revision never becomes current output.

Results that are degraded (`FitResult`, `RepairResult`, `GateDecision`) carry a reason; failures are not silently replaced by apparently successful output.

## Worker model

- **API process**: asynchronous, owns admission and the control plane. Run a single instance for now (filesystem jobs and in-memory guards).
- **Dub work**: admitted chunk processes, scheduled by focus (what the viewer is watching), lookahead and missing work. Cross-process limits cap concurrent ASR, MT and TTS calls.
- **Lip-sync worker**: one persistent process per GPU holding LatentSync and SyncNet in memory. It claims candidates with a lock, sends heartbeats, prioritises the chunk the viewer is watching, and publishes atomically.
- **Locks**: each lip-sync chunk is claimed with an exclusive lock file carrying a host-pid-time token and refreshed by a heartbeat; a lock that goes stale is taken over atomically, so several GPUs and machines can share the queue. (Fencing tokens are on the roadmap.)
- **Processes**: child processes run in their own process groups so timeouts and cancellation kill the whole group; orphans are swept on startup.

## Revisions, fencing and restart

Edits, undo and expression changes create new revisions in an append-only store kept separate from job snapshots. Operation records plus reconciliation tie the two together; there is no multi-file transaction. Idempotency keys make duplicate submissions safe. On restart the server reconciles interrupted edits, reclaims expired leases and resumes jobs that still need work.

## Live controls

Buttons, transcript edits and the chat planner all go through `ActionService.submit()`. LLM tool calls are treated as untrusted input: arguments are validated against shared schemas, authorised, and bound to expiring confirmation tokens where needed. Missing control authentication disables control startup; it never bypasses authentication. Adding a language mid-watch schedules the chunks ahead of the playback position first (the runway), then the rest.

## Status of the extraction

What is in the repository today:

- **Implemented and tested**: `core/` (typed settings and legacy env mapping), `clients/`, `pipeline/` (Stage protocol, runner,
  `DubWorkflow`, typed `Segment`, one module per stage), `planning/` (planner split into focused modules plus the `voices/`
  package with `VoicePlanner`), `jobs/scheduler/` (add-language scheduler), `control/` (actions, chat, expression, revision
  store, guards), `server/` (app factory and container, routers, services, file repository), `lipsync/`, `workers/` and
  `mirrors/`.
- **Not built yet**: `storage/` (artifact manifests and cache), `workers/leases.py` fencing tokens, `jobs/state.py` transition
  rules and the `language/` modules beyond number normalisation. The sections above that describe them are the target
  behaviour; the on-disk formats are unchanged from the prototype.
- **Moved, not yet rewritten**: the stage implementations (`stages/tts.py`, `mix.py`, `speakers.py`, `asr.py`, ...) and
  `planning/planner.py` and `fitting/fit_stage.py` still contain long functions from the prototype. Golden tests for the TTS
  stage pin their behaviour for the planned split into `SegmentSynthesizer`, `SentenceSplitter` and `RunawayRepairLadder`.
- **Relaxed rules**: modules listed in the `[tool.ruff.lint.per-file-ignores]` and `[[tool.mypy.overrides]]` blocks of
  `pyproject.toml` are exempt from strict lint and typing until they are rewritten. Everything else is strict.
