from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from src.simulation.backtester import Backtester


class DummyPyFunc:
    """A minimal stand-in for ``mlflow.pyfunc.PyFuncModel`` used in tests."""

    def predict(self, df):  # noqa: D401
        # Always return FLAT (1) action to keep logic deterministic
        return np.array([1])


@pytest.mark.unit
def test_backtester_runs(monkeypatch):
    """Backtester should complete without errors and return sane outputs."""

    # ---------------- Mock out mlflow ----------------
    import mlflow.pyfunc as _mlpyfunc

    monkeypatch.setattr(_mlpyfunc, "load_model", lambda uri: DummyPyFunc())

    # ---------------- Dummy data ----------------
    df = pd.DataFrame({
        "close": np.linspace(1, 10, 10),
        "feature1": np.random.randn(10),
    })

    bt = Backtester(model_uri="dummy:/model", data=df, initial_balance=100.0)

    result = bt.run()

    # Equity curve length should match number of dataframe rows (since env terminates before last step).
    assert len(result.equity_curve) == len(df)
    # No trades expected because model always returns FLAT
    assert result.trades.empty
    # Metrics should contain keys
    assert set(result.metrics) >= {"total_profit", "sharpe_ratio", "max_drawdown"} 