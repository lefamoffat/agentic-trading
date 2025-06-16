import pytest
import numpy as np
import pandas as pd

from src.callbacks.utils import (
    get_annualization_factor,
    calculate_trade_metrics,
    calculate_performance_metrics,
)
from src.environments.trading_env import Trade
from src.environments.types import Position

@pytest.mark.unit
class TestCallbackUtils:
    """Test suite for callback utility functions."""

    @pytest.mark.parametrize("timeframe, expected_factor", [
        ("1h", 252 * 24),
        ("4h", 252 * 6),
        ("1d", 252),
        ("15m", 252 * 24 * 4),
        ("unsupported_format", 252), # Test default case
    ])
    def test_get_annualization_factor(self, timeframe, expected_factor):
        """Test the annualization factor calculation for different timeframes."""
        assert get_annualization_factor(timeframe) == expected_factor

    def test_calculate_trade_metrics(self):
        """Test calculation of metrics from trade history."""
        # Scenario 1: No trades
        metrics = calculate_trade_metrics([])
        assert metrics["total_trades"] == 0
        assert metrics["win_rate_pct"] == 0.0
        assert metrics["profit_factor"] == 0.0

        # Scenario 2: A mix of winning and losing trades
        trades = [
            Trade(entry_price=1, exit_price=2, position=Position.LONG, profit=100),
            Trade(entry_price=2, exit_price=1, position=Position.SHORT, profit=50),
            Trade(entry_price=3, exit_price=2, position=Position.LONG, profit=-75),
        ]
        metrics = calculate_trade_metrics(trades)
        assert metrics["total_trades"] == 3
        assert metrics["win_rate_pct"] == pytest.approx((2 / 3) * 100)
        assert metrics["profit_factor"] == pytest.approx(150 / 75)
        
        # Scenario 3: Only winning trades (should not result in division by zero)
        winning_trades = [trades[0], trades[1]]
        metrics = calculate_trade_metrics(winning_trades)
        assert metrics["profit_factor"] == 0.0 # Implementation returns 0 when losses are zero
        
        # Scenario 4: Only losing trades
        losing_trades = [trades[2]]
        metrics = calculate_trade_metrics(losing_trades)
        assert metrics["profit_factor"] == 0.0

    def test_calculate_performance_metrics(self):
        """Test calculation of comprehensive performance metrics."""
        # Scenario 1: Steady portfolio growth, no drawdown
        portfolio_values = np.linspace(10000, 12000, 100).tolist()
        trades = [Trade(1, 2, Position.LONG, 1000), Trade(2, 1, Position.SHORT, 1000)]
        
        metrics = calculate_performance_metrics(portfolio_values, trades, timeframe="1d")
        
        expected_keys = [
            "sharpe_ratio", "sortino_ratio", "calmar_ratio", "profit_pct",
            "max_drawdown_pct", "win_rate_pct", "profit_factor", "total_trades"
        ]
        assert all(key in metrics for key in expected_keys)

        assert metrics["total_trades"] == 2
        assert metrics["win_rate_pct"] == 100.0
        assert metrics["profit_pct"] == pytest.approx(20.0)
        assert metrics["max_drawdown_pct"] == 0.0

        # Scenario 2: Portfolio with drawdown
        portfolio_values_dd = [100, 110, 105, 120, 90, 115]
        metrics_dd = calculate_performance_metrics(portfolio_values_dd, [], timeframe="1h")
        assert metrics_dd["max_drawdown_pct"] < 0
        assert metrics_dd["sharpe_ratio"] != 0
        assert metrics_dd["sortino_ratio"] != 0
        assert metrics_dd["calmar_ratio"] != 0

    def test_calculate_performance_metrics_no_data(self):
        """Test behavior with insufficient data."""
        metrics = calculate_performance_metrics([], [], "1d")
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["profit_pct"] == 0.0
        assert metrics["total_trades"] == 0
        
        metrics_short = calculate_performance_metrics([100], [], "1d")
        assert metrics_short["sharpe_ratio"] == 0.0 