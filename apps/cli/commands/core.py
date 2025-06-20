from __future__ import annotations

"""Core / miscellaneous CLI commands (project init, validation, diagnostics)."""

import platform
import subprocess
import sys
from pathlib import Path

import typer
from mlflow.tracking import MlflowClient
import mlflow

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
# MLflow server management
# ---------------------------------------------------------------------------


@app.command("mlflow-start")
def mlflow_start(port: int = typer.Option(5001, help="Port to run MLflow server on")) -> None:
    """Start MLflow tracking server in Docker."""
    
    launch_script = SCRIPTS_DIR / "setup" / "launch_mlflow.sh"
    if not launch_script.exists():
        typer.secho(f"❌ Launch script not found: {launch_script}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    
    typer.echo(f"🚀 Starting MLflow server on port {port}...")
    _run(["/bin/bash", str(launch_script), str(port)])
    typer.secho(f"✅ MLflow server started on http://localhost:{port}", fg=typer.colors.GREEN)


@app.command("mlflow-stop")
def mlflow_stop() -> None:
    """Stop MLflow tracking server."""
    
    try:
        result = subprocess.run(
            ["docker", "stop", "mlflow_server"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            typer.secho("✅ MLflow server stopped", fg=typer.colors.GREEN)
        else:
            typer.secho("⚠️  No MLflow server container found or already stopped", fg=typer.colors.YELLOW)
            
    except subprocess.TimeoutExpired:
        typer.secho("❌ Timeout stopping MLflow server", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"❌ Error stopping MLflow server: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command("mlflow-status")
def mlflow_status() -> None:
    """Check MLflow server status."""
    
    # Check if container is running
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=mlflow_server", "--format", "table {{.Names}}\t{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if "mlflow_server" in result.stdout:
            typer.secho("🐳 MLflow Docker container: RUNNING", fg=typer.colors.GREEN)
        else:
            typer.secho("🐳 MLflow Docker container: NOT RUNNING", fg=typer.colors.YELLOW)
            
    except Exception as e:
        typer.secho(f"❌ Error checking Docker status: {e}", fg=typer.colors.RED)
    
    # Check if MLflow API is reachable
    try:
        from src.utils.mlflow import _is_mlflow_reachable, _tracking_uri
        
        if _is_mlflow_reachable():
            typer.secho(f"🌐 MLflow API: REACHABLE at {_tracking_uri()}", fg=typer.colors.GREEN)
        else:
            typer.secho(f"🌐 MLflow API: NOT REACHABLE at {_tracking_uri()}", fg=typer.colors.RED)
            
    except Exception as e:
        typer.secho(f"❌ Error checking MLflow API: {e}", fg=typer.colors.RED)


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
        MlflowClient().search_registered_models()
        typer.secho("✅ MLflow reachable", fg=typer.colors.GREEN)
    except (mlflow.MlflowException, ConnectionError) as exc:
        typer.secho(f"❌ MLflow not reachable: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc 