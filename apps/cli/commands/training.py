from __future__ import annotations

"""Training-related commands using the background training service."""

import asyncio
import uuid
from datetime import datetime
from typing import Optional
import subprocess
import json
import sys
import os
import time

import typer

from apps.cli import app
from src.training import get_training_service
from src.messaging import TrainingStatus
from src.utils.logger import get_logger

# Rich progress bar for nicer live output
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

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
    experiment_id = f"train_{symbol.replace('/', '')}_{timeframe}_{uuid.uuid4().hex[:8]}"
    config = {
        "agent_type": agent_type,
        "symbol": symbol,
        "timeframe": timeframe,
        "timesteps": timesteps,
        "learning_rate": learning_rate,
        "initial_balance": initial_balance,
    }
    typer.echo(f"🚀 Starting training experiment: {experiment_id}")
    typer.echo(f"📊 Symbol: {symbol}, Timeframe: {timeframe}")
    typer.echo(f"🤖 Agent: {agent_type}, Timesteps: {timesteps:,}")
    
    # Launch the worker as a background process
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "worker.log")
    log_file = open(log_path, "a")
    config_json = json.dumps(config)
    subprocess.Popen(
        [sys.executable, "scripts/training/run_training_worker.py", experiment_id, config_json],
        stdout=log_file,
        stderr=log_file,
        close_fds=True,
    )
    typer.echo("✅ Training process launched in background!")
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
        try:
            training_service = await get_training_service()
            
            if experiment_id:
                # Show specific experiment status
                result = await training_service.get_experiment_status(experiment_id)
                
                if result["status"] == "not_found":
                    typer.echo(f"❌ Experiment not found: {experiment_id}")
                    return
                
                # Display detailed status
                typer.echo(f"📊 Experiment Status: {experiment_id}")
                typer.echo(f"🔄 Status: {result['status']}")
                typer.echo(f"📈 Progress: {result['current_step']:,} / {result['total_steps']:,} ({result['progress']:.1%})")
                typer.echo(f"⏱️  Duration: {result['duration']:.1f} seconds")
                
                if result.get('metrics'):
                    typer.echo("📊 Latest Metrics:")
                    for key, value in result['metrics'].items():
                        if isinstance(value, float):
                            typer.echo(f"   {key}: {value:.4f}")
                        else:
                            typer.echo(f"   {key}: {value}")
            else:
                # List all experiments
                experiments = await training_service.list_experiments()
                
                if not experiments:
                    typer.echo("📭 No training experiments found")
                    return
                
                typer.echo(f"📋 Training Experiments ({len(experiments)} total):")
                typer.echo("")
                
                for exp in experiments[-10:]:  # Show last 10
                    exp_id = exp.get("experiment_id", "unknown")
                    status = exp.get("status", "unknown")
                    config = exp.get("config", {})
                    symbol = config.get("symbol", "N/A")
                    agent = config.get("agent_type", "N/A")
                    
                    # Status emoji
                    status_emoji = {
                        "running": "🔄",
                        "completed": "✅", 
                        "failed": "❌",
                        "cancelled": "⏹️",
                        "starting": "🚀"
                    }.get(status, "❓")
                    
                    typer.echo(f"{status_emoji} {exp_id[:16]}... | {symbol} | {agent} | {status}")
                
        except Exception as e:
            typer.echo(f"❌ Failed to check status: {e}", err=True)
            raise typer.Exit(1)
    
    asyncio.run(_check_status())

@app.command()
def stop(
    experiment_id: str = typer.Argument(..., help="Experiment ID to stop")
) -> None:
    """Stop a running training experiment."""
    async def _stop_training():
        try:
            training_service = await get_training_service()
            result = await training_service.stop_experiment(experiment_id)
            
            if result["status"] == "not_found":
                typer.echo(f"❌ Experiment not found: {experiment_id}")
                return
            
            typer.echo(f"⏹️  Stopped experiment: {experiment_id}")
            typer.echo(f"📈 Status: {result['status']}")
            
        except Exception as e:
            typer.echo(f"❌ Failed to stop experiment: {e}", err=True)
            raise typer.Exit(1)
    
    asyncio.run(_stop_training())

@app.command("list")
def list_experiments() -> None:
    """List all training experiments."""
    async def _list_experiments():
        try:
            training_service = await get_training_service()
            experiments = await training_service.list_experiments()
            
            if not experiments:
                typer.echo("📭 No training experiments found")
                return
            
            typer.echo(f"📋 All Training Experiments ({len(experiments)} total):")
            typer.echo("")
            typer.echo("ID                                | Symbol    | Agent | Status    | Progress")
            typer.echo("-" * 85)
            
            for exp in experiments:
                exp_id = exp.get("experiment_id", exp.get("id", "unknown"))  # Try both fields
                config = exp.get("config", {})
                symbol = config.get("symbol", "N/A")[:8]
                agent = config.get("agent_type", "N/A")[:5]
                status = exp.get("status", "unknown")[:9]
                
                current_step = exp.get("current_step", 0)
                total_steps = exp.get("total_steps", 0)
                progress = f"{current_step:,}/{total_steps:,}" if total_steps > 0 else "N/A"
                
                typer.echo(f"{exp_id:<35} | {symbol:<8} | {agent:<5} | {status:<9} | {progress}")
                
        except Exception as e:
            typer.echo(f"❌ Failed to list experiments: {e}", err=True)
            raise typer.Exit(1)
    
    asyncio.run(_list_experiments())

async def _watch_experiment(experiment_id: str, interval: int = 5) -> None:
    """Monitor experiment progress in real-time."""
    training_service = await get_training_service()
    
    console = Console()
    console.rule(f"[bold blue]Watching {experiment_id}")
    
    # Wait briefly for experiment registration (handles race right after launch)
    start_wait = time.monotonic()
    first = await training_service.get_experiment_status(experiment_id)
    while first["status"] == "not_found" and (time.monotonic() - start_wait) < 15:
        console.print("Waiting for experiment registration…", style="yellow")
        await asyncio.sleep(2)
        first = await training_service.get_experiment_status(experiment_id)

    if first["status"] == "not_found":
        typer.echo(f"❌ Experiment not found: {experiment_id} (after waiting 15s)")
        return

    total_steps: int | None = first.get("total_steps")
    if not total_steps or total_steps <= 0:
        # Fallback to percentage-based bar (0-100)
        total_steps = 100
        percentage_mode = True
    else:
        percentage_mode = False

    progress_display = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.1f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=4,
    )

    with progress_display:
        task_id = progress_display.add_task("Training", total=total_steps)

        try:
            while True:
                result = await training_service.get_experiment_status(experiment_id)

                if result["status"] == "not_found":
                    console.print(f"[red]❌ Experiment not found: {experiment_id}")
                    break

                # Determine completion value
                if percentage_mode:
                    completed = min(100, max(0, result["progress"] * 100))
                else:
                    completed = result.get("current_step", 0)
                progress_display.update(task_id, completed=completed)

                # Show key metrics below the progress bar
                metrics = result.get("metrics", {})
                if metrics:
                    metrics_str = " | ".join(
                        f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                        for k, v in list(metrics.items())[:5]
                    )
                    console.print(f"[cyan]{metrics_str}", justify="left")

                if result["status"] in ["completed", "failed", "cancelled"]:
                    console.print(f"\n🏁 Training finished with status: [bold]{result['status']}[/bold]")
                    break

                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            console.print("\n⏹️  Detached from experiment – it keeps running in the background.")

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