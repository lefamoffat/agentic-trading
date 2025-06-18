from __future__ import annotations

"""Agentic Trading unified CLI.

Usage examples::

    # project setup
    agentic init

    # build features
    agentic build-features --symbol EUR/USD --timeframe 1h

    # train agent
    agentic train --symbol EUR/USD --timeframe 1h --timesteps 20000

    # run the Streamlit simulator
    agentic simulate

    # do everything with sensible defaults
    agentic quickstart
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})


def _run(cmd: list[str] | str, *, cwd: Optional[Path] = None) -> None:
    """Run a subprocess and forward stdout/stderr."""

    if isinstance(cmd, str):
        cmd_list = cmd if sys.platform == "win32" else cmd.split()
    else:
        cmd_list = cmd

    # Always execute through uv run
    if cmd_list[0] == sys.executable:
        cmd_list = ["uv", "run"] + cmd_list

    typer.echo(f"🚀 Running: {' '.join(cmd_list)}", err=False)
    completed = subprocess.run(cmd_list, cwd=cwd or PROJECT_ROOT)
    if completed.returncode != 0:
        typer.echo("❌ Command failed.", err=True)
        raise typer.Exit(code=completed.returncode)


# ---------------------------------------------------------------------------
# Validate configuration helper
# ---------------------------------------------------------------------------

def _load_config(path: Path | None = None):  # noqa: D401
    """Load AppConfig or exit on failure."""
    try:
        from src.utils.config import app_config  # import here to avoid heavy deps at CLI startup

        return app_config  # Singleton loads on first access
    except Exception as exc:  # pylint: disable=broad-except
        typer.echo(f"❌ Config validation failed → {exc}", err=True)
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def init() -> None:
    """Initialize the project directory structure and test configs."""

    script = SCRIPTS_DIR / "setup" / "init_project.py"
    _run([sys.executable, str(script)])


@app.command("build-features")
def build_features(
    symbol: str = typer.Option("EUR/USD", help="Trading symbol e.g. EUR/USD"),
    timeframe: str = typer.Option("1h", help="Timeframe e.g. 1h, 1d"),
) -> None:
    """Generate feature CSVs using Qlib."""

    script = SCRIPTS_DIR / "features" / "build_features.py"
    _run([sys.executable, str(script), "--symbol", symbol, "--timeframe", timeframe])


@app.command()
def train(
    symbol: str = typer.Option("EUR/USD", help="Trading symbol e.g. EUR/USD"),
    timeframe: str = typer.Option("1h", help="Timeframe e.g. 1h, 1d"),
    timesteps: int = typer.Option(20_000, help="Total training timesteps"),
    agent: str = typer.Option("ppo", help="Agent name (ppo, dqn, a2c, etc.)"),
) -> None:
    """Train an RL agent and log the run to MLflow."""

    script = SCRIPTS_DIR / "training" / "train_agent.py"
    _run(
        [
            sys.executable,
            str(script),
            "--symbol",
            symbol,
            "--timeframe",
            timeframe,
            "--timesteps",
            str(timesteps),
            "--agent",
            agent,
        ]
    )


@app.command()
def optimize(
    symbol: str = typer.Option("EUR/USD", help="Trading symbol"),
    timeframe: str = typer.Option("1h", help="Timeframe"),
    timesteps: int = typer.Option(5_000, help="Timesteps per trial"),
    trials: int = typer.Option(20, help="Number of Optuna trials"),
    agent: str = typer.Option("ppo", help="Agent name"),
) -> None:
    """Run Optuna hyperparameter optimization."""

    script = SCRIPTS_DIR / "training" / "optimize_agent.py"
    _run(
        [
            sys.executable,
            str(script),
            "--symbol",
            symbol,
            "--timeframe",
            timeframe,
            "--timesteps",
            str(timesteps),
            "--trials",
            str(trials),
            "--agent",
            agent,
        ]
    )


@app.command()
def simulate() -> None:
    """Launch the Streamlit simulation dashboard."""
    script = SCRIPTS_DIR / "analysis" / "run_simulation.py"
    # Use Streamlit CLI to run the script so hot-reload works
    _run(["streamlit", "run", str(script)])


@app.command()
def quickstart(
    symbol: str = typer.Option("EUR/USD", help="Trading symbol"),
    timeframe: str = typer.Option("1h", help="Timeframe"),
    timesteps: int = typer.Option(20_000, help="Training timesteps"),
    agent: str = typer.Option("ppo", help="Agent name"),
) -> None:
    """Run init → build-features → train → simulate with sensible defaults.

    All steps use the provided ``symbol``, ``timeframe``, etc., so you can tweak
    them in one place.
    """

    init()
    prepare_data(symbol=symbol, timeframe=timeframe)
    train(symbol=symbol, timeframe=timeframe, timesteps=timesteps, agent=agent)
    simulate()


# ---------------------------------------------------------------------------
# New commands: validate-config & doctor
# ---------------------------------------------------------------------------

@app.command("validate-config")
def validate_config() -> None:
    """Validate YAML configuration files via Pydantic."""

    _load_config()
    typer.secho("✅ Configuration is valid", fg=typer.colors.GREEN)


@app.command()
def doctor() -> None:
    """Run environment diagnostics (UV, MLflow, Python, config)."""

    import platform
    from mlflow.tracking import MlflowClient

    typer.echo("ℹ️  Environment diagnostics\n----------------------")
    typer.echo(f"Python:     {platform.python_version()}")
    typer.echo(f"Platform:   {platform.platform()}")

    # UV presence
    if Path(sys.executable).name != "uv":
        typer.secho("⚠️  Python executable is not 'uv'. Make sure to run via 'uv run'.", fg=typer.colors.YELLOW)
    else:
        typer.secho("✅ Using uv Python wrapper", fg=typer.colors.GREEN)

    # Config validation
    try:
        _load_config()
        typer.secho("✅ Config validated", fg=typer.colors.GREEN)
    except typer.Exit:  # validation already prints error
        pass

    # MLflow connection test
    try:
        MlflowClient().list_registered_models()  # quick call
        typer.secho("✅ MLflow reachable", fg=typer.colors.GREEN)
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"❌ MLflow not reachable: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# Data download & preparation commands
# ---------------------------------------------------------------------------

from src.types import Timeframe  # noqa: E402  # Imported late to avoid heavy deps at startup


@app.command("download-data")
def download_data(
    symbol: str = typer.Option("EUR/USD", help="Trading symbol e.g. EUR/USD"),
    timeframe: str = typer.Option("1h", help="Timeframe e.g. 1h, 1d"),
    bars: int = typer.Option(365, help="Number of bars to fetch"),
) -> None:
    """Download raw historical data for a symbol & timeframe using the dedicated script."""

    script = SCRIPTS_DIR / "data" / "download_historical.py"
    _run(
        [
            sys.executable,
            str(script),
            "--symbol",
            symbol,
            "--timeframe",
            timeframe,
            "--bars",
            str(bars),
        ]
    )


@app.command("prepare-data")
def prepare_data(
    symbol: str | None = typer.Option(None, help="Trading symbol e.g. EUR/USD"),
    timeframe: str | None = typer.Option(None, help="Timeframe e.g. 1h, 1d"),
    bars: int = typer.Option(365, help="Number of bars to fetch"),
) -> None:
    """Run the full data preparation pipeline → download → dump → build-features.

    If ``symbol``/``timeframe`` are omitted they will be requested interactively.
    """

    # Interactive prompts when values are omitted
    if symbol is None:
        symbol = typer.prompt("Enter trading symbol", default="EUR/USD")
    if timeframe is None:
        timeframe = typer.prompt("Enter timeframe (e.g. 1h, 1d)", default="1h")

    # 1) Download raw data ---------------------------------------------------
    download_data(symbol=symbol, timeframe=timeframe, bars=bars)  # reuse impl above

    # 2) Dump to Qlib binary format -----------------------------------------
    try:
        freq = Timeframe.from_standard(timeframe).qlib_name
    except ValueError as exc:  # noqa: BLE001
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    csv_path = f"data/qlib_source/{timeframe}"
    dump_script = SCRIPTS_DIR / "data" / "dump_bin.py"
    _run(
        [
            sys.executable,
            str(dump_script),
            "dump_all",
            "--csv-path",
            csv_path,
            "--qlib-dir",
            "data/qlib_data",
            "--freq",
            freq,
            "--include_fields",
            "open,high,low,close,volume,factor",
            "--date_field_name",
            "date",
        ]
    )

    # 3) Build engineered features ------------------------------------------
    build_features(symbol=symbol, timeframe=timeframe)


if __name__ == "__main__":
    app() 