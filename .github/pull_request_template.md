## Summary

<!-- What does this change and why? Link the issue. -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor (no behaviour change)
- [ ] Behaviour change (outputs differ)
- [ ] Docs / CI / tooling

## Behaviour impact

<!-- Does this change what the pipeline produces (thresholds, prompts, ordering, retries, mixing, gates)? If yes, say so explicitly and describe the difference. -->

## Validation evidence

- [ ] `make check` passes locally (ruff, mypy, pytest)
- [ ] Added or updated tests; for behaviour-sensitive changes, a fixture and golden file
- [ ] If a GPU run was needed, describe what was run and the result

## Configuration and docs

- [ ] New settings are documented (`.env.example`, `docs/configuration.md`) with default, disabled behaviour and restart requirement
- [ ] Docs updated where relevant
- [ ] `THIRD_PARTY.md` updated if dependencies or models changed

## Hygiene

- [ ] No secrets, cookies, weights, media or real job output committed
- [ ] No imports with side effects (downloads, model loads, env parsing)
