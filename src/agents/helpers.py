#!/usr/bin/env python3
"""MLflow integration helpers for agents."""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["build_observation"]


def build_observation(model_input: pd.DataFrame) -> np.ndarray:
    """Transform MLflow PyFunc DataFrame input to SB3 observation format.
    
    This function is part of the MLflow model serving pipeline that converts
    pandas DataFrame input (MLflow's standard serving format) into numpy
    observation arrays that Stable-Baselines3 models expect.
    
    Args:
        model_input: DataFrame with market data containing any market features
        
    Returns:
        Observation array matching TradingEnv observation space format
        Shape: (batch_size, N) for batch input or (N,) for single observation
        where N = market_features + 4 portfolio + 9 time features
        
    Raises:
        ValueError: If input data is invalid
    """
    # Get batch size
    batch_size = len(model_input)
    observations = []
    
    # Define metadata columns to exclude from market features
    metadata_columns = {'timestamp', 'symbol', 'date'}
    
    # Get market feature columns (exclude metadata)
    market_columns = [col for col in model_input.columns if col not in metadata_columns]
    
    # If no market columns, use OHLCV fallback
    if not market_columns:
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        market_features = []
        for col in ohlcv_columns:
            if col in model_input.columns:
                market_features.append(model_input[col].values.astype(np.float32))
            else:
                # Missing OHLCV column, fill with zeros
                market_features.append(np.zeros(batch_size, dtype=np.float32))
        
        if market_features:
            market_obs = np.column_stack(market_features)
        else:
            # No market data at all, create zero features
            market_obs = np.zeros((batch_size, 5), dtype=np.float32)
    else:
        # Use available market columns as features
        market_features = []
        for col in market_columns:
            values = model_input[col].values.astype(np.float32)
            market_features.append(values)
        
        market_obs = np.column_stack(market_features)
    
    observations.append(market_obs)
    
    # Portfolio features (4 features) - serving defaults for inference
    portfolio_obs = np.column_stack([
        np.ones(batch_size, dtype=np.float32),    # balance_normalized = 1.0
        np.ones(batch_size, dtype=np.float32),    # position_type = 1 (FLAT)
        np.zeros(batch_size, dtype=np.float32),   # entry_price_normalized = 0.0
        np.zeros(batch_size, dtype=np.float32)    # unrealized_pnl_pct = 0.0
    ])
    observations.append(portfolio_obs)
    
    # Time features (9 features) - serving defaults for inference
    time_obs = np.column_stack([
        np.ones(batch_size, dtype=np.float32),    # market_open = 1.0
        np.ones(batch_size, dtype=np.float32) * 0.5,  # time_of_day_normalized = 0.5
        np.ones(batch_size, dtype=np.float32),    # day_monday = 1.0
        np.zeros(batch_size, dtype=np.float32),   # day_tuesday = 0.0
        np.zeros(batch_size, dtype=np.float32),   # day_wednesday = 0.0
        np.zeros(batch_size, dtype=np.float32),   # day_thursday = 0.0
        np.zeros(batch_size, dtype=np.float32),   # day_friday = 0.0
        np.zeros(batch_size, dtype=np.float32),   # day_saturday = 0.0
        np.zeros(batch_size, dtype=np.float32),   # day_sunday = 0.0
    ])
    observations.append(time_obs)
    
    # Combine observations
    observation = np.concatenate(observations, axis=1)
    
    # Remove batch dimension for single example
    if batch_size == 1:
        observation = observation[0]
    
    return observation 