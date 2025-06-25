from __future__ import annotations

"""Core / miscellaneous CLI commands (project init, validation, diagnostics)."""

import platform
import subprocess
import sys
import signal
import os
from pathlib import Path
from typing import Optional

import typer

from apps.cli import app, _run, _load_config, PROJECT_ROOT, SCRIPTS_DIR

# ML tracking integration
try:
    from src.tracking import get_ml_tracker, get_experiment_repository
    TRACKING_AVAILABLE = True
except ImportError:
    TRACKING_AVAILABLE = False

# ---------------------------------------------------------------------------
# Project initialisation
# ---------------------------------------------------------------------------

@app.command()
def init() -> None:
    """Create the standard directory structure and sanity-check configs."""

    script = SCRIPTS_DIR / "setup" / "init_project.py"
    _run([sys.executable, str(script)])

# ---------------------------------------------------------------------------
# ML tracking backend management
# ---------------------------------------------------------------------------

@app.command("tracking-status")
def tracking_status():
    """Check ML tracking backend status."""
    if not TRACKING_AVAILABLE:
        typer.echo("❌ ML tracking not available")
        return
    
    try:
        import asyncio
        
        async def check_status():
            tracker = await get_ml_tracker()
            repository = await get_experiment_repository()
            health = await repository.get_system_health()
            
            if health.is_healthy:
                typer.echo(f"✅ ML tracking backend is healthy")
                typer.echo(f"📊 Total experiments: {health.total_experiments}")
                typer.echo(f"🏃 Active experiments: {health.active_experiments}")
                typer.echo(f"📈 Total runs: {health.total_runs}")
            else:
                typer.echo(f"❌ ML tracking backend is unhealthy")
                if health.error_message:
                    typer.echo(f"Error: {health.error_message}")
        
        asyncio.run(check_status())
        
    except Exception as e:
        typer.echo(f"❌ Failed to check ML tracking status: {e}")

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
    """Run environment diagnostics (Python, uv, ML tracking, config)."""

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

    # ML tracking availability
    try:
        import asyncio

        async def check_health():
            from src.tracking import get_experiment_repository
            repository = await get_experiment_repository()
            health = await repository.get_system_health()
            return health.is_healthy, health.error_message

        healthy, error_msg = asyncio.run(check_health())
        if healthy:
            typer.secho("✅ ML tracking reachable", fg=typer.colors.GREEN)
        else:
            typer.secho("❌ ML tracking backend unhealthy", fg=typer.colors.RED)
            if error_msg:
                typer.echo(f"Error: {error_msg}")
            raise typer.Exit(code=2)

    except Exception as exc:
        typer.secho(f"❌ ML tracking not reachable: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc

    # All checks passed
    typer.secho("🎉 All diagnostics passed", fg=typer.colors.GREEN)
    raise typer.Exit(code=0)

# System status commands
@app.command("status")
def system_status():
    """Check overall system status."""
    typer.echo("🔍 Checking system status...")
    
    # Check Python environment
    typer.echo(f"🐍 Python: {sys.version.split()[0]}")
    
    # Check ML tracking
    if TRACKING_AVAILABLE:
        try:
            import asyncio
            
            async def check_tracking():
                repository = await get_experiment_repository()
                health = await repository.get_system_health()
                
                if health.is_healthy:
                    typer.echo(f"✅ ML tracking: healthy ({health.total_experiments} experiments)")
                else:
                    typer.echo(f"❌ ML tracking: unhealthy")
            
            asyncio.run(check_tracking())
        except Exception as e:
            typer.echo(f"❌ ML tracking: error - {e}")
    else:
        typer.echo("❌ ML tracking not available")
    
    # Check if in project directory
    if Path("pyproject.toml").exists():
        typer.echo("✅ Project environment detected")
    else:
        typer.echo("❌ Not in project root directory")

@app.command("version")
def version():
    """Show version information."""
    try:
        # Try to get version from pyproject.toml
        import tomllib
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
            version = data.get("project", {}).get("version", "unknown")
            typer.echo(f"Agentic Trading System v{version}")
    except Exception:
        typer.echo("Agentic Trading System (version unknown)")

if __name__ == "__main__":
    app() 