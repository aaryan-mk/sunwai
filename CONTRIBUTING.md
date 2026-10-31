# Contributing to SunwAI

Thank you for helping. This guide covers setup, standards and what a good pull request contains.

## Development setup

```bash
git clone https://github.com/aaryan-mk/sunwai.git
cd sunwai
uv sync --extra dev --frozen     # CPU-only: no GPU packages, no weights
pre-commit install
make check                       # ruff, mypy, pytest
```

The `dev` extra never installs GPU packages or downloads model weights. You do not need a GPU, Bodhan key or OpenRouter key to run the default tests; they use fakes and recorded fixtures.

## Workflow

1. Open an issue for anything larger than a small fix, so the approach can be agreed first.
2. Branch from `main`; keep one logical change per pull request.
3. Run `make check` before pushing. CI runs the same checks.
4. Fill in the pull request template.

## Standards

- **Tooling**: Ruff (lint and format; do not also run Black), mypy, pytest.
- **Types**: `mypy --strict` for new and extracted modules. Narrow, documented overrides only for untyped vendor code; no blanket `ignore_errors`.
- **Size**: files target 150-300 lines (review above 400); functions target 40 lines or fewer (review above 60). Split by responsibility.
- **Naming**: domain words with units, such as `slot_duration_s`, `playback_ms`, `sample_rate_hz`. No `utils.py`.
- **Docstrings**: Google style; state contracts, units, side effects and invariants.
- **Data**: no untyped segment dictionaries in new code. Frozen dataclasses for domain records; Pydantic at HTTP, configuration and persisted-JSON boundaries.
- **Errors**: typed domain errors, translated to HTTP only in `server/`. Use `raise ... from exc`. Retry only classified transient failures, bounded by attempts and deadline.
- **Imports**: no downloads, environment parsing, model loads, threads or CUDA initialisation at import time. GPU dependencies are optional and imported lazily.
- **Logging**: structured fields (`job_id`, `language`, `chunk`, `segment`, `revision`, `stage`, `attempt`, `duration_ms`). Never log secrets or full transcripts by default.
- **Pure logic**: decisions take arrays, records and config and return values; filesystem, HTTP, clocks and subprocesses live in adapters.

## Evidence for behaviour-sensitive changes

If a change can alter what the pipeline produces (thresholds, prompts, ordering, retry budgets, mixing, gates), include fixture-based evidence: a recorded chunk under `tests/fixtures/` and a golden file under `tests/golden/`. Compare text and JSON exactly; compare audio by samples and duration with a stated tolerance, never by encoded MP4 bytes. State clearly in the pull request when a change alters outputs rather than merely moving code.

## Configuration

Every new setting needs: a named field with a default, a description, its disabled behaviour, cache impact, scope and whether it needs a restart. Add it to `.env.example` and `docs/configuration.md`. `scripts/check_config_inventory.py` fails on undocumented settings.

## Tests

- `tests/unit`: pure logic. `tests/contracts`: HTTP clients, API schemas, disk formats, CLI.
- `tests/integration`: CPU subprocess, locking, restart and publication (needs ffmpeg).
- `tests/gpu`: opt-in model quality and performance; never required by CI.

Do not commit model weights, downloaded videos, cookies, credentials or real job output. Test fixtures must be small and redistributable.

## Licence of contributions

By contributing you agree your work is licensed under the Apache License 2.0 (see `LICENSE`). Be mindful of third-party licences when adding dependencies and update `THIRD_PARTY.md`.

## Conduct and security

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md), not in public issues.
