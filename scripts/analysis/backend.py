"""Headless backend helper for the interactive simulation UI.

Isolated from Streamlit so it can be imported and tested without pulling
in heavy UI dependencies.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlflow
import pandas as pd

from src.simulation import Backtester, SimulationResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=32)
def _download_model_artifact(model_uri: str) -> str:
    """Download the underlying SB3 model ZIP and return local path.

    This thin wrapper exists primarily to benefit from functools caching.
    """
    logger.info(f"Ensuring local copy of model {model_uri}…")
    return mlflow.artifacts.download_artifacts(model_uri)


def run_simulation(
    model_uri: str,
    data_path: str,
    initial_balance: float = 10_000.0,
) -> SimulationResult:
    """High-level orchestration function used by Streamlit UI.

    Args:
        model_uri: A valid MLflow model URI (e.g. "models:/my-model@v2").
        data_path: Path to a CSV file with engineered features identical to
            the model's training data.
        initial_balance: Starting cash balance.

    Returns:
        :class:`SimulationResult` instance with trade log & metrics.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not locate features file: {data_path}")

    df = pd.read_csv(data_path)
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])  # keep numeric index for env

    # Remove rows with any NaNs to prevent invalid observations producing
    # NaN logits during policy prediction.
    if df.isna().any().any():
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        logger.warning("Dropped %d rows with NaNs from features file.", before - len(df))

    backtester = Backtester(model_uri=model_uri, data=df, initial_balance=initial_balance)
    return backtester.run()
