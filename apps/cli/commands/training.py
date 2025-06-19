from __future__ import annotations

"""Training-related commands (train agent, hyperparameter optimisation)."""

import sys
from pathlib import Path

import typer

from apps.cli import app, _run, SCRIPTS_DIR

# ---------------------------------------------------------------------------
# Train agent
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Hyperparameter optimisation
# ---------------------------------------------------------------------------


@app.command()
def optimize(
    symbol: str = typer.Option("EUR/USD", help="Trading symbol"),
    timeframe: str = typer.Option("1h", help="Timeframe"),
    timesteps: int = typer.Option(5_000, help="Timesteps per trial"),
    trials: int = typer.Option(20, help="Number of Optuna trials"),
    agent: str = typer.Option("ppo", help="Agent name"),
) -> None:
    """Run Optuna HPO for the specified agent."""

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