from __future__ import annotations

"""Simulation / orchestration commands (Streamlit dashboard, quickstart)."""

import sys
import typer

from apps.cli import app, _run, SCRIPTS_DIR
from apps.cli.commands import data as data_cmd
from apps.cli.commands import training as training_cmd
from apps.cli.commands import core as core_cmd

# ---------------------------------------------------------------------------
# Streamlit simulation / dashboard
# ---------------------------------------------------------------------------


@app.command()
def simulate() -> None:
    """Launch the (legacy) Streamlit simulation UI.

    This will be migrated to the multipage dashboard in Phase-5 but is kept for
    backwards compatibility until that work lands.
    """

    script = SCRIPTS_DIR / "analysis" / "run_simulation.py"
    _run(["streamlit", "run", str(script)])


# ---------------------------------------------------------------------------
# Quickstart – full pipeline in one command
# ---------------------------------------------------------------------------


@app.command()
def quickstart(
    symbol: str = typer.Option("EUR/USD", help="Trading symbol"),
    timeframe: str = typer.Option("1h", help="Timeframe"),
    timesteps: int = typer.Option(20_000, help="Training timesteps"),
    agent: str = typer.Option("ppo", help="Agent name"),
) -> None:
    """End-to-end demo: init → data → train → simulate."""

    # 1) project init
    core_cmd.init()

    # 2) data prep
    data_cmd.prepare_data(symbol=symbol, timeframe=timeframe)

    # 3) training
    training_cmd.train(symbol=symbol, timeframe=timeframe, timesteps=timesteps, agent=agent)

    # 4) simulation
    simulate() 