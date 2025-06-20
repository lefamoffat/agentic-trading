#!/usr/bin/env python3
"""MLflow integration helpers for agents."""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["build_observation"]


def build_observation(model_input: pd.DataFrame) -> np.ndarray:
    """Convert feature dataframe into observation tensor for SB3 policies.

    Dynamically builds observations to match the environment's CompositeObservation:
    - Dynamic market features (any columns in input DataFrame)
    - Portfolio features (4): [balance_norm, position_type, entry_price_norm, unrealized_pnl_pct]  
    - Time features (9): [market_open, time_of_day_normalized, day_monday, ..., day_sunday]

    Args:
        model_input: DataFrame from feature pipeline.

    Returns:
        NumPy array with shape (batch, n_features) and dtype float32.
    """
    batch_size = len(model_input)
    observations = []
    
    # Market features (dynamic from DataFrame columns)
    market_columns = [col for col in model_input.columns 
                     if col not in ['timestamp', 'symbol', 'date']]
    
    if market_columns:
        market_obs = model_input[market_columns].to_numpy(dtype=np.float32)
        observations.append(market_obs)
    else:
        # Fallback: basic OHLCV
        basic_features = ["close", "volume", "high", "low", "open"]
        market_features = []
        for col in basic_features:
            if col in model_input.columns:
                values = model_input[col].to_numpy(dtype=np.float32)
            else:
                values = np.zeros(batch_size, dtype=np.float32)
            market_features.append(values)
        market_obs = np.column_stack(market_features)
        observations.append(market_obs)
    
    # Portfolio features (4 features) - serving defaults
    portfolio_obs = np.column_stack([
        np.ones(batch_size, dtype=np.float32),    # balance_normalized = 1.0
        np.ones(batch_size, dtype=np.float32),    # position_type = 1 (FLAT)
        np.zeros(batch_size, dtype=np.float32),   # entry_price_normalized = 0.0
        np.zeros(batch_size, dtype=np.float32)    # unrealized_pnl_pct = 0.0
    ])
    observations.append(portfolio_obs)
    
    # Time features (9 features) - serving defaults
    time_obs = np.column_stack([
        np.ones(batch_size, dtype=np.float32),    # market_open = 1.0
        np.full(batch_size, 0.5, dtype=np.float32),  # time_of_day_normalized = 0.5
        # Day encoding (assume Monday)
        np.ones(batch_size, dtype=np.float32),   # day_monday = 1.0
        np.zeros(batch_size, dtype=np.float32),  # day_tuesday = 0.0
        np.zeros(batch_size, dtype=np.float32),  # day_wednesday = 0.0
        np.zeros(batch_size, dtype=np.float32),  # day_thursday = 0.0
        np.zeros(batch_size, dtype=np.float32),  # day_friday = 0.0
        np.zeros(batch_size, dtype=np.float32),  # day_saturday = 0.0
        np.zeros(batch_size, dtype=np.float32)   # day_sunday = 0.0
    ])
    observations.append(time_obs)
    
    # Combine observations
    observation = np.concatenate(observations, axis=1)
    
    # Remove batch dimension for single example
    if batch_size == 1:
        observation = observation[0]
    
    return observation 