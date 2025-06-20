#!/usr/bin/env python3
"""Utilities for custom callbacks.
"""
from typing import Dict, List

import numpy as np
import pandas as pd

from src.environment.state.position import Trade
from src.types.enums import Timeframe


def get_annualization_factor(timeframe: str) -> int:
    """Get the annualization factor based on the data timeframe.
    Assumes 252 trading days per year.
    """
    if 'h' in timeframe:
        hours = int(timeframe.replace('h', ''))
        return 252 * (24 // hours)
    if 'd' in timeframe:
        return 252
    if 'm' in timeframe:
        minutes = int(timeframe.replace('m', ''))
        return 252 * 24 * (60 // minutes)
    return 252 # Default for daily

def calculate_trade_metrics(trade_history: List[Trade]) -> Dict[str, float]:
    """Calculate metrics from a list of trades."""
    if not trade_history:
        return {
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
        }

    profits = np.array([trade.profit for trade in trade_history if trade.profit > 0])
    losses = np.array([abs(trade.profit) for trade in trade_history if trade.profit < 0])

    total_trades = len(trade_history)
    winning_trades = len(profits)
    win_rate_pct = (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0

    total_profit = np.sum(profits)
    total_loss = np.sum(losses)
    profit_factor = total_profit / total_loss if total_loss > 0 else 0.0

    return {
        "win_rate_pct": win_rate_pct,
        "profit_factor": profit_factor,
        "total_trades": float(total_trades),
    }


def calculate_performance_metrics(
    portfolio_values: List[float],
    trade_history: List[Trade],
    timeframe: str
) -> Dict[str, float]:
    """Calculate performance metrics from portfolio values and trade history.
    """
    trade_metrics = calculate_trade_metrics(trade_history)

    if not portfolio_values or len(portfolio_values) < 2:
        base_metrics = {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "profit_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }
        return {**base_metrics, **trade_metrics}

    returns = pd.Series(portfolio_values).pct_change().dropna()
    annualization_factor = get_annualization_factor(timeframe)

    # Sharpe Ratio
    sharpe_ratio = 0.0
    if returns.std() != 0:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(annualization_factor)

    # Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    sortino_ratio = 0.0
    if downside_std != 0:
        sortino_ratio = (returns.mean() / downside_std) * np.sqrt(annualization_factor)

    # Total Profit
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    profit_pct = ((final_value - initial_value) / initial_value) * 100

    # Max Drawdown
    cumulative_max = pd.Series(portfolio_values).cummax()
    drawdown = (pd.Series(portfolio_values) - cumulative_max) / cumulative_max
    max_drawdown_pct = drawdown.min() * 100

    # Calmar Ratio
    calmar_ratio = 0.0
    if max_drawdown_pct != 0:
        # Calculate the precise number of days in the evaluation period
        timeframe_enum = Timeframe.from_standard(timeframe)
        total_minutes = timeframe_enum.minutes * len(portfolio_values)
        total_days = total_minutes / (60 * 24)

        if total_days > 0:
            total_return = (final_value - initial_value) / initial_value
            # Use 365.25 for a more accurate annualization over leap years
            annualized_return = (1 + total_return) ** (365.25 / total_days) - 1
            calmar_ratio = annualized_return / abs(max_drawdown_pct / 100)

    performance_metrics = {
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "profit_pct": profit_pct,
        "max_drawdown_pct": max_drawdown_pct,
    }

    return {**performance_metrics, **trade_metrics}
