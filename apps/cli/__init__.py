from __future__ import annotations

"""Agentic Trading – modular CLI (apps/cli).

This package exposes the ``agentic`` console-script.  All sub-commands are
implemented in dedicated modules under ``apps.cli.commands`` to keep the code
base maintainable as the project grows.
"""

from pathlib import Path
import subprocess
import sys
from typing import Optional

import typer

# ---------------------------------------------------------------------------
# Paths & globals
# ---------------------------------------------------------------------------

# …/repo/apps/cli/__init__.py → repo root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"

# Typer root application
app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _run(cmd: list[str] | str, *, cwd: Optional[Path] = None) -> None:  # noqa: D401
    """Run a subprocess under ``uv run`` and stream output.

    Raises
    ------
    typer.Exit
        If the command returns a non-zero exit code.
    """

    # Accept both list and string syntax for convenience
    if isinstance(cmd, str):
        cmd_list = cmd.split() if sys.platform != "win32" else [cmd]
    else:
        cmd_list = cmd

    # Always execute Python scripts through *uv* for consistent environment
    if cmd_list[0] == sys.executable:
        cmd_list = ["uv", "run"] + cmd_list

    typer.echo(f"🚀 Running: {' '.join(cmd_list)}")
    result = subprocess.run(cmd_list, cwd=cwd or PROJECT_ROOT)
    if result.returncode != 0:
        typer.secho("❌ Command failed.", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=result.returncode)


def _load_config(_path: Path | None = None):  # noqa: D401, ANN001
    """Import the *AppConfig* singleton or abort with a helpful error."""

    try:
        from src.utils.config import app_config  # local import to avoid heavy deps at startup

        return app_config
    except Exception as exc:  # pylint: disable=broad-except
        typer.secho(f"❌ Config validation failed → {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# Register command modules (import side-effects)
# ---------------------------------------------------------------------------

# The imports below attach functions to *app* via decorators inside the modules.
from apps.cli.commands import core as _core  # noqa: F401  pylint: disable=wrong-import-position
from apps.cli.commands import data as _data  # noqa: F401  pylint: disable=wrong-import-position
from apps.cli.commands import training as _training  # noqa: F401  pylint: disable=wrong-import-position