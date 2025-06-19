from __future__ import annotations

"""Core / miscellaneous CLI commands (project init, validation, diagnostics)."""

import platform
import sys
from pathlib import Path

import typer
from mlflow.tracking import MlflowClient

from apps.cli import app, _run, _load_config, PROJECT_ROOT, SCRIPTS_DIR

# ---------------------------------------------------------------------------
# Project initialisation
# ---------------------------------------------------------------------------


@app.command()
def init() -> None:
    """Create the standard directory structure and sanity-check configs."""

    script = SCRIPTS_DIR / "setup" / "init_project.py"
    _run([sys.executable, str(script)])


# ---------------------------------------------------------------------------
# Config validation & diagnostics
# ---------------------------------------------------------------------------


@app.command("validate-config")
def validate_config() -> None:  # noqa: D401
    """Ensure YAML config files pass Pydantic validation."""

    _load_config()
    typer.secho("✅ Configuration is valid", fg=typer.colors.GREEN)


@app.command()
def doctor() -> None:
    """Run environment diagnostics (Python, uv, MLflow, config)."""

    typer.echo("ℹ️  Environment diagnostics\n----------------------")
    typer.echo(f"Python:     {platform.python_version()}")
    typer.echo(f"Platform:   {platform.platform()}")

    # uv presence
    if Path(sys.executable).name != "uv":
        typer.secho("⚠️  Python executable is not 'uv'. Consider running via 'uv run'.", fg=typer.colors.YELLOW)
    else:
        typer.secho("✅ Using uv Python wrapper", fg=typer.colors.GREEN)

    # Config validation
    try:
        _load_config()
        typer.secho("✅ Config validated", fg=typer.colors.GREEN)
    except typer.Exit:
        return

    # MLflow availability
    try:
        MlflowClient().list_registered_models()
        typer.secho("✅ MLflow reachable", fg=typer.colors.GREEN)
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"❌ MLflow not reachable: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc 