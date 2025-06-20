#!/usr/bin/env python3
"""Centralised helper functions that wrap MLflow calls.

The goal is to isolate MLflow-specific logic so that the rest of the
codebase never needs to import the mlflow package directly.  This makes
unit-testing easier (one place to monkey-patch) and cushions us from
future MLflow API changes.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional
import subprocess
import time

import mlflow
import pandas as pd
from mlflow import ActiveRun
from mlflow.models import infer_signature
import os

from urllib.parse import urljoin

import requests

from src.utils.exceptions import ConfigurationError, TradingSystemError
from src.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "log_metrics",
    "log_params",
    "log_sb3_model",
    "start_experiment_run",
    "ensure_mlflow_running",
]

# ---------------------------------------------------------------------------
# Global helper (single source of truth)
# ---------------------------------------------------------------------------


def _tracking_uri() -> str:
    """Return the effective MLflow tracking URI.

    Defaults to ``http://127.0.0.1:5001`` if the env var is not set.
    """
    return os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")


def _is_mlflow_reachable() -> bool:  # pragma: no cover
    """Return True if a server is listening at the tracking URI."""
    uri = _tracking_uri()

    try:
        resp = requests.get(urljoin(uri + "/", "api/2.0/mlflow/experiments/list"), timeout=5)
        if resp.ok:
            mlflow.set_tracking_uri(uri)
        return True
    except (requests.RequestException, requests.Timeout):
        pass

    return False


def _start_mlflow_server() -> bool:
    """Start MLflow server using Docker if not already running.
    
    Returns:
        True if server started successfully, False otherwise
    """
    try:
        # Get project root (assuming we're in src/utils/)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        launch_script = os.path.join(project_root, "scripts", "setup", "launch_mlflow.sh")
        
        if not os.path.exists(launch_script):
            logger.error(f"MLflow launch script not found: {launch_script}")
            return False
        
        logger.info("Starting MLflow server via Docker...")
        
        # Run the launch script
        result = subprocess.run(
            ["/bin/bash", launch_script],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            logger.error(f"Failed to start MLflow server: {result.stderr}")
            return False
        
        # Wait for server to be ready (up to 30 seconds)
        for i in range(30):
            time.sleep(1)
            if _is_mlflow_reachable():
                logger.info(f"MLflow server started successfully on {_tracking_uri()}")
                return True
            if i % 5 == 0:  # Log every 5 seconds
                logger.info(f"Waiting for MLflow server to be ready... ({i+1}s)")
            
        logger.error("MLflow server failed to become ready within 30 seconds")
        return False
        
    except subprocess.TimeoutExpired:
        logger.error("MLflow server startup timed out")
        return False
    except Exception as e:
        logger.error(f"Error starting MLflow server: {e}")
        return False


def ensure_mlflow_running() -> None:
    """Ensure MLflow server is running, auto-starting if necessary.
    
    Raises:
        ConfigurationError: If MLflow cannot be started or reached
    """
    if _is_mlflow_reachable():
        logger.debug("MLflow server is already running")
        return
    
    logger.info("MLflow server not reachable, attempting to start...")
    
    if not _start_mlflow_server():
        raise ConfigurationError(
            f"Failed to start MLflow server. Please ensure Docker is running and try again. "
            f"Alternatively, start MLflow manually or set MLFLOW_TRACKING_URI to a running instance."
        )


def ensure_experiment(experiment_name: str) -> str:  # pragma: no cover
    """Return the experiment id, creating the experiment if it doesn't exist.

    Raises:
        ConfigurationError: If MLflow tracking URI is unreachable
    """
    # Auto-start MLflow if needed
    ensure_mlflow_running()

    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = client.create_experiment(name=experiment_name)
    else:
        exp_id = exp.experiment_id

    mlflow.set_experiment(experiment_name)
    return exp_id

# ---------------------------------------------------------------------------
# Experiment / run helpers
# ---------------------------------------------------------------------------


@contextmanager
def start_experiment_run(
    run_name: str,
    experiment_name: str | None = None,
    tags: Optional[Dict[str, str]] = None,
) -> Generator[ActiveRun, None, None]:
    """Context-manager that starts (and closes) an MLflow run.

    Args:
        run_name: Friendly name shown in the MLflow UI.
        experiment_name: If provided, `mlflow.set_experiment()` is called
            before starting the run.
        tags: Arbitrary metadata tags.

    Yields:
        The active :class:`mlflow.ActiveRun` instance.

    """
    if experiment_name is not None:
        ensure_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name, tags=tags) as run:  # type: ignore[arg-type]
        yield run


# ---------------------------------------------------------------------------
# Thin wrappers around common logging calls (makes stubbing trivial)
# ---------------------------------------------------------------------------


def log_params(params: Dict[str, Any]) -> None:
    """Shorthand for ``mlflow.log_params`` that silently ignores empty dicts."""
    if params:
        mlflow.log_params(params)


def log_metrics(metrics: Dict[str, float], step: int | None = None) -> None:
    if not metrics:
        return

    cleaned: Dict[str, float] = {}
    for k, v in metrics.items():
        try:
            val = float(v)
        except (TypeError, ValueError):
            continue  # skip non numeric

        if val != val or val in (float("inf"), float("-inf")):
            # NaN or ±Inf → skip to avoid MLflow UI crash
            continue
        cleaned[k] = val

    if cleaned:
        mlflow.log_metrics(cleaned, step=step)


# ---------------------------------------------------------------------------
# Model logging
# ---------------------------------------------------------------------------


def log_sb3_model(
    model,  # Stable-Baselines3 BaseAlgorithm - kept untyped to avoid heavy import.
    name: str,
    artifacts: Dict[str, str],
    signature_df: pd.DataFrame,
    python_model,
) -> str:
    """Log an SB3 model as a PyFunc and return its model URI.

    Args:
        model: Trained SB3 agent (only used for saving - not strictly
            required but kept for symmetry).
        name: Logical name for the logged model (shown under **Models** in
            MLflow UI).
        artifacts: ``{"model_path": "/abs/path/to/file.zip"}`` - must at
            minimum contain the saved SB3 `model_path`.
        signature_df: Representative dataframe used **only** for signature
            inference. Passed through :pyfunc:`mlflow.models.infer_signature`.
        python_model: An instance of :class:`mlflow.pyfunc.PythonModel` (our
            wrapper) that knows how to load and predict.

    Returns:
        The model URI (string) of the newly logged model.

    """
    # New in MLflow 3 - no *artifact_path* argument; *name* is required.
    logged_model = mlflow.pyfunc.log_model(
        name=name,
        python_model=python_model,
        artifacts=artifacts,
        input_example=signature_df,
        signature=infer_signature(signature_df),
    )

    # Explicitly register the model so it shows up in the Model Registry.
    try:
        mlflow.register_model(model_uri=logged_model.model_uri, name=name)
    except mlflow.MlflowException as exc:  # pragma: no cover
        # Registration may fail if the model already exists or registry not configured.
        import warnings

        warnings.warn(f"[mlflow] Model registration skipped: {exc}")

    return logged_model.model_uri

# ---------------------------------------------------------------------------
# Helper constants
# ---------------------------------------------------------------------------
