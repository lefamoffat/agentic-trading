from __future__ import annotations

"""Training-related commands using the background training service."""

import asyncio
from typing import Optional, Any, Dict, List
import json

import typer
from pydantic import BaseModel, Field

from apps.cli import app
from apps.cli.api_client import post, get as http_get, ws as ws_connect
from src.utils.logger import get_logger

# Rich progress bar for nicer live output
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

# Third-party WS exceptions for graceful error handling
from websockets.exceptions import InvalidStatus, ConnectionClosed

logger = get_logger(__name__)

@app.command()
def train(
    symbol: str = typer.Option("EUR/USD", help="Trading symbol e.g. EUR/USD"),
    timeframe: str = typer.Option("1h", help="Timeframe e.g. 1h, 1d"),
    timesteps: int = typer.Option(20_000, help="Total training timesteps"),
    agent_type: str = typer.Option("ppo", help="RL agent type: ppo, a2c, sac, dqn"),
    learning_rate: float = typer.Option(0.0003, help="Learning rate"),
    initial_balance: float = typer.Option(10000.0, help="Initial trading balance"),
    watch: bool = typer.Option(
        True,
        "--watch/--no-watch",
        help="Attach to live progress after launching experiment",
    ),
) -> None:
    """Start a new training experiment and optionally attach to live progress."""
    config = {
        "agent_type": agent_type,
        "symbol": symbol,
        "timeframe": timeframe,
        "timesteps": timesteps,
        "learning_rate": learning_rate,
        "initial_balance": initial_balance,
    }

    typer.echo(f"🚀 Launching training via API: {symbol} {timeframe} » {agent_type}")

    async def _launch():
        resp = await post("/experiments", json=config)
        if resp.status_code not in (202, 200):
            typer.echo(f"❌ Failed to launch experiment: {resp.text}", err=True)
            raise typer.Exit(1)
        data = resp.json()
        return data["experiment_id"]

    try:
        experiment_id = asyncio.run(_launch())
    except Exception as exc:
        typer.echo(f"❌ Error communicating with API: {exc}", err=True)
        raise typer.Exit(1)

    typer.echo("✅ Training request accepted by API!")
    typer.echo(f"📝 Experiment ID: {experiment_id}")

    if watch:
        typer.echo("👀 Switching to live watch (Ctrl+C to detach)…")
        try:
            asyncio.run(_watch_experiment(experiment_id))
        except KeyboardInterrupt:
            typer.echo("\n⏹️  Detached from live watch – experiment continues in background.")
    else:
        typer.echo(f"💡 Run 'agentic watch {experiment_id}' to follow progress")

@app.command()
def status(
    experiment_id: Optional[str] = typer.Argument(None, help="Experiment ID to check")
) -> None:
    """Check the status of training experiments."""
    async def _check_status():
        if experiment_id:
            resp = await http_get(f"/experiments/{experiment_id}")
            if resp.status_code == 404:
                typer.echo(f"❌ Experiment not found: {experiment_id}")
                return
            resp.raise_for_status()
            raw_result = resp.json()

            experiment = _ExperimentSummaryPayload.model_validate(raw_result, strict=True)

            typer.echo(f"📊 Experiment Status: {experiment.experiment_id}")
            typer.echo(f"🔄 Status: {experiment.status}")
            typer.echo(
                f"📈 Progress: {experiment.current_step:,} / {experiment.total_steps:,} "
                f"({(experiment.current_step / experiment.total_steps * 100) if experiment.total_steps else 0:.1f}%)",
            )

            metrics = raw_result.get("metrics", {})
            if metrics:
                typer.echo("📊 Latest Metrics:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        typer.echo(f"   {key}: {value:.4f}")
                    else:
                        typer.echo(f"   {key}: {value}")
        else:
            resp = await http_get("/experiments", params={"limit": 100})
            resp.raise_for_status()
            raw_experiments: List[Dict[str, Any]] = resp.json()

            # Strict conversion ------------------------------------------------
            experiments: List[_ExperimentSummaryPayload] = [
                _ExperimentSummaryPayload.model_validate(item, strict=True) for item in raw_experiments
            ]

            if not experiments:
                typer.echo("📭 No training experiments found")
                return

            typer.echo(f"📋 Training Experiments ({len(experiments)} total):\n")
            for experiment in experiments:
                experiment_id = experiment.experiment_id
                status = experiment.status

                symbol = experiment.config.symbol[:8]
                agent_type = experiment.config.agent_type[:5]

                status_emoji = {
                    "running": "🔄",
                    "completed": "✅",
                    "failed": "❌",
                    "cancelled": "⏹️",
                    "starting": "🚀",
                }.get(status, "❓")
                typer.echo(f"{status_emoji} {experiment_id[:16]}... | {symbol} | {agent_type} | {status}")

    try:
        asyncio.run(_check_status())
    except Exception as exc:
        typer.echo(f"❌ Failed to check status: {exc}", err=True)
        raise typer.Exit(1)

@app.command()
def stop(
    experiment_id: str = typer.Argument(..., help="Experiment ID to stop")
) -> None:
    """Stop a running training experiment."""
    async def _stop_training():
        try:
            resp = await post(f"/experiments/{experiment_id}/stop")
            if resp.status_code == 404:
                typer.echo(f"❌ Experiment not found: {experiment_id}")
                return
            resp.raise_for_status()
            typer.echo(f"⏹️  Stop requested: {experiment_id}")
            typer.echo(f"📈 Status: {resp.json().get('status')}")
        except Exception as e:
            typer.echo(f"❌ Failed to stop experiment: {e}", err=True)
            raise typer.Exit(1)
    
    asyncio.run(_stop_training())

@app.command("list")
def list_experiments() -> None:
    """List all training experiments."""
    async def _list_experiments():
        try:
            resp = await http_get("/experiments", params={"limit": 1000})
            resp.raise_for_status()
            raw_experiments: List[Dict[str, Any]] = resp.json()
            
            # Strict conversion ------------------------------------------------
            experiments: List[_ExperimentSummaryPayload] = [
                _ExperimentSummaryPayload.model_validate(item, strict=True) for item in raw_experiments
            ]
            
            if not experiments:
                typer.echo("📭 No training experiments found")
                return
            
            typer.echo(f"📋 All Training Experiments ({len(experiments)} total):")
            typer.echo("")
            typer.echo("ID                                | Symbol    | Agent | Status    | Progress")
            typer.echo("-" * 85)
            
            for experiment in experiments:
                experiment_id = experiment.experiment_id
                status = experiment.status

                symbol = experiment.config.symbol[:8]
                agent_type = experiment.config.agent_type[:5]
                
                current_step = experiment.current_step
                total_steps = experiment.total_steps
                progress = f"{current_step:,}/{total_steps:,}" if total_steps > 0 else "N/A"
                
                typer.echo(
                    f"{experiment_id:<35} | {symbol:<8} | {agent_type:<5} | {status:<9} | {progress}"
                )
                
        except Exception as e:
            typer.echo(f"❌ Failed to list experiments: {e}", err=True)
            raise typer.Exit(1)
    
    asyncio.run(_list_experiments())

async def _watch_experiment(experiment_id: str, interval: int = 5) -> None:
    """Watch experiment progress in real-time via WebSocket."""
    console = Console()

    progress = Progress(
        TextColumn("[bold blue]Training:"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TextColumn("Step {task.fields[step]}/{task.fields[total_steps]}", justify="right"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[metrics]}", justify="left"),
        console=console,
        transient=True,
    )

    task_id = progress.add_task("", total=100, step=0, total_steps=0, metrics="")

    with progress:
        try:
            async with ws_connect(f"/ws/experiments/{experiment_id}") as websocket:
                async for raw in websocket:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    if msg.get("topic") == "training.progress":
                        data = msg["data"]
                        current = data.get("current_step", 0)
                        total = data.get("total_steps", 1)
                        percent = current / total * 100 if total else 0
                        progress.update(task_id, completed=percent, step=current, total_steps=total)

                    elif msg.get("topic") == "training.metrics":
                        metrics = msg["data"].get("metrics", {})
                        metrics_str = " | ".join(
                            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                            for k, v in list(metrics.items())[:3]
                        )
                        progress.update(task_id, metrics=metrics_str)

                    elif msg.get("topic") == "training.status":
                        status = msg["data"].get("status")
                        if status in ("completed", "failed", "cancelled"):
                            progress.update(task_id, metrics=f"status: {status}")
                            break
        except (InvalidStatus, ConnectionClosed, OSError) as exc:
            # Gracefully degrade when WS connection fails – inform the user and exit
            console.print(
                f"[red]❌ Real-time updates unavailable ({exc}). "
                f"Use 'agentic status {experiment_id}' to poll progress.[/red]",
            )
            return

@app.command("watch")
def watch(
    experiment_id: str = typer.Argument(..., help="Experiment ID to monitor"),
    interval: int = typer.Option(5, help="Update interval in seconds")
) -> None:
    """Attach to a running experiment and stream live progress and metrics."""
    asyncio.run(_watch_experiment(experiment_id, interval))

# Backwards-compatibility alias – hidden from help output
@app.command("monitor", hidden=True)
def monitor_alias(
    experiment_id: str = typer.Argument(...),
    interval: int = typer.Option(5)
) -> None:  # pragma: no cover
    """[Deprecated] Use 'watch' instead."""
    asyncio.run(_watch_experiment(experiment_id, interval))

# ---------------------------------------------------------------------------
# Payload models – convert raw JSON responses into objects for attribute
# access.  The CLI intentionally fails fast if required fields are missing.
# ---------------------------------------------------------------------------

class _ExperimentConfigPayload(BaseModel):
    """Subset of experiment configuration returned by the API."""

    symbol: str
    agent_type: str

class _ExperimentSummaryPayload(BaseModel):
    """Minimal payload used by the CLI list / status commands."""

    experiment_id: str
    status: str
    current_step: int = Field(..., ge=0)
    total_steps: int = Field(..., ge=0)
    config: _ExperimentConfigPayload

    # Ignore any additional keys from the API – the CLI does not need them

    model_config = {
        "extra": "ignore",
    }