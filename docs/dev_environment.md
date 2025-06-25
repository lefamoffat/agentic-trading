# Developer Environment with Podman

This guide explains how to spin up the external services (Redis & Aim) required for local development and testing without Docker, using **Podman**.

## Prerequisites

1. **Podman** ≥ 4.3 installed (`brew install podman` or your package manager).
2. Python 3.11 + `uv` or Poetry for dependency management (see `README.md`).

## Quick start

```bash
# create and activate virtual-env, install project (with dev extras)
uv pip install -e '.[dev]'  # uses dependencies from pyproject.toml

# start services
./scripts/podman/start_redis.sh          # starts Redis on :6379
./scripts/podman/start_aim.sh            # starts Aim server on :43800

# run the smoke test (verifies the whole stack)
uv run scripts/dev/smoke_training_test.py
```

After development, stop the containers:

```bash
./scripts/podman/stop_redis.sh
./scripts/podman/stop_aim.sh
```

The scripts are idempotent and can be executed multiple times; they will skip starting if a container is already running.

## Continuous usage

Add these aliases to your shell profile:

```bash
alias redis-up="scripts/podman/start_redis.sh"
alias redis-down="scripts/podman/stop_redis.sh"
alias aim-up="scripts/podman/start_aim.sh"
alias aim-down="scripts/podman/stop_aim.sh"
```

## Troubleshooting

-   **Podman not running?** On macOS you need the Podman VM: `podman machine init && podman machine start`.
-   **Port already in use?** Pass a custom port: `./scripts/podman/start_redis.sh 6380`.
-   **Aim UI not reachable?** Visit `http://localhost:43800` in your browser; wait a few seconds after startup.
-   **Image pull failed?** If Podman reports a 401/403 pulling `aimstack/aim-server`, ensure you're on `main` and re-run the script – we now use the public `aimstack/aim` image.

Happy hacking! :rocket:
