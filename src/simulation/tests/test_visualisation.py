from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.simulation.visualisation import equity_curve_figure, price_with_trades


@pytest.mark.unit
def test_equity_curve_figure_creation():
    series = pd.Series(np.linspace(100, 110, 10))
    fig = equity_curve_figure(series)
    assert isinstance(fig, go.Figure)
    # Should contain one trace
    assert len(fig.data) == 1


@pytest.mark.unit
def test_price_with_trades_creation():
    price_series = pd.Series(np.linspace(1, 10, 10))
    trades_df = pd.DataFrame(
        {
            "entry_price": [2.0, 5.0],
            "exit_price": [3.0, 6.0],
            "position": ["long", "short"],
            "profit": [1.0, -1.0],
        }
    )

    fig = price_with_trades(price_series, trades_df)
    assert isinstance(fig, go.Figure)
    # Expect at least price line + 4 markers = 5 traces
    assert len(fig.data) >= 5 