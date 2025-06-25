# Implementation Roadmap 2024-06

This document tracks the _forward-looking_ work. Completed historical tasks were removed for clarity.

## Phase 0 Groundwork (local)

| Item                  | Deliverable                                                                                                                                                                                     |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0.1 Tooling baseline  | • `.pre-commit-config.yaml` with ruff-lint, ruff-format, black, mypy, pytest hooks.<br>• `pyproject.toml` already pins Python 3.11 and adds ruff/mypy in `[project.optional-dependencies.dev]`. |
| 0.2 Debug-print purge | Remove `print()` calls from library code; replace with `logger.debug/info` as appropriate.                                                                                                      |
| 0.3 Dead-test purge   | Delete/adjust tests that reference removed APIs (`_publish_progress`, legacy callbacks, etc.).                                                                                                  |
| 0.4 Smoke test script | `scripts/dev/smoke_training_test.py` – 20-step training run that asserts Redis counters advance & Aim run created. Executed manually for now.                                                   |

## Phase 1 Static safety-nets

1. Enable strict ruff (`DUP`, `T201` disallow print, etc.) and mypy.
2. Add duplicate-code / dead-code linters.
3. Pydantic schema for `ExperimentState`; Redis broker serialises via this model.

## Phase 2 Podman dev services

Podman helper scripts to start/stop Redis & Aim locally; documented in `docs/dev_environment.md`.

## Phase 3 Integration tests

Pytest fixtures start Redis/Aim via Podman; smoke test becomes automated `@pytest.mark.integration`.

## Phase 4 CI with Podman

GitHub Actions (or other) running the full pipeline: lint → static analysis → integration tests with Podman containers.

## Phase 5 Refactors for maintainability

Split `TrainingCallback`, unify event-loop handling, centralise error handling.

## Phase 6 Observability

Structured JSON logging, health-probe script.

## Phase 7 Hardening (stretch)

Mutation testing, perf benchmarks, ADR docs.

---

_progress_:

-   Phase 0 ✅ Groundwork complete – tooling baseline in place; debug prints & dead tests removed.
-   Phase 1 ✅ Static safety nets – strict ruff/mypy enabled; duplicate/dead-code detection active; `ExperimentState` Pydantic model integrated into Redis broker.
-   Phase 2 ✅ Podman dev services – helper scripts for Redis & Aim and documentation added.
-   Phase 3 ✅ Integration tests – Podman-backed Redis fixture and smoke integration test automated.
-   Phase 4 ✅ CI – workflow runs lint, type-check, unit & integration tests inside containers.
-   Phase 5 🟡 Refactor – callback split (`metric_aggregator`, `progress_dispatcher`, `SB3TrainingCallback`), single event-loop handling, `async_guard` decorator live. Remaining:
    • align unit tests with `SB3TrainingCallback` (in-progress)
    • apply `async_guard` to other background coroutines (in-progress)
    • polish documentation / remove stray legacy imports
    • ensure ruff/mypy pass after cleanup
