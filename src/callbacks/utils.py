#!/usr/bin/env python3
"""
Utilities for custom callbacks.
"""
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

def calculate_performance_metrics(portfolio_values: List[float]) -> Dict[str, float]:
    """
    Calculate performance metrics from a list of portfolio values.
    
    Args:
        portfolio_values (List[float]): A list of portfolio values over an episode.
        
    Returns:
        Dict[str, float]: A dictionary of calculated metrics.
    """
    if not portfolio_values:
        return {
            "sharpe_ratio": 0.0,
            "profit_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }

    returns = pd.Series(portfolio_values).pct_change().dropna()
    
    # Sharpe Ratio (annualized, assuming daily data)
    sharpe_ratio = 0.0
    if returns.std() != 0:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        
    # Total Profit
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    profit_pct = ((final_value - initial_value) / initial_value) * 100
    
    # Max Drawdown
    cumulative_max = pd.Series(portfolio_values).cummax()
    drawdown = (pd.Series(portfolio_values) - cumulative_max) / cumulative_max
    max_drawdown_pct = drawdown.min() * 100
    
    return {
        "sharpe_ratio": sharpe_ratio,
        "profit_pct": profit_pct,
        "max_drawdown_pct": max_drawdown_pct,
    }

def get_episode_portfolio_values(infos: List[Dict[str, Any]]) -> List[float]:
    """
    Extract portfolio values from the info dictionaries of an episode.
    
    The Monitor wrapper logs the final info of an episode. If we want step-by-step
    values, we need to collect them manually.
    """
    # This is a placeholder; the actual collection will happen in a custom wrapper.
    # For now, we assume the final portfolio value is in the 'info' dict.
    portfolio_values = [info.get("portfolio_value") for info in infos if "portfolio_value" in info]
    return portfolio_values 