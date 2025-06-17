#!/usr/bin/env python3
"""Stateless helper utilities specific to SB3 models."""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["build_observation"]


def build_observation(model_input: pd.DataFrame) -> np.ndarray:
    """Convert a feature dataframe into an observation tensor for SB3 policies.

    The observation layout mirrors the custom TradingEnv used in this
    project:

    ``[close_price, balance_norm (1.0), position (1=FLAT), *features]``

    Args:
        model_input: DataFrame produced by the feature pipeline.

    Returns:
        NumPy array with shape ``(batch, 3 + n_features)`` and dtype
        ``float32``.

    """
    features = model_input.to_numpy(dtype=np.float32)

    if "close" in model_input.columns:
        close_vals = model_input["close"].to_numpy(dtype=np.float32).reshape(-1, 1)
    else:
        close_vals = features[:, [0]]

    balance_vals = np.ones((features.shape[0], 1), dtype=np.float32)
    position_vals = np.ones((features.shape[0], 1), dtype=np.float32)

    return np.concatenate([close_vals, balance_vals, position_vals, features], axis=1)
