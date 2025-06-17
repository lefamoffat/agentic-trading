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

import mlflow
import pandas as pd
from mlflow import ActiveRun
from mlflow.models import infer_signature

__all__ = [
    "log_metrics",
    "log_params",
    "log_sb3_model",
    "start_experiment_run",
]

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
        mlflow.set_experiment(experiment_name)

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
    if metrics:
        mlflow.log_metrics(metrics, step=step)


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
    return logged_model.model_uri
