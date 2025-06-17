"""Plotly figure builders for simulation results."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

__all__ = [
    "price_with_trades",
    "equity_curve_figure",
]


def price_with_trades(
    price_series: pd.Series,
    trades: pd.DataFrame,
    title: str | None = None,
) -> go.Figure:
    """Create a candlestick/line chart with buy/sell markers.

    Args:
        price_series: Series of prices indexed by step or datetime.
        trades: DataFrame returned by :class:`Backtester` containing at
            minimum `entry_price`, `exit_price`, and `position` columns.
        title: Optional chart title.

    Returns:
        A :class:`plotly.graph_objects.Figure` instance.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=price_series.index,
            y=price_series.values,
            mode="lines",
            name="Price",
            line={"color": "#1f77b4"},
        )
    )

    # Add trade markers
    for _, trade in trades.iterrows():
        pos = trade["position"].lower()
        marker_symbol = "triangle-up" if pos == "long" else "triangle-down" if pos == "short" else "circle"
        marker_color = "green" if pos == "long" else "red" if pos == "short" else "gray"

        fig.add_trace(
            go.Scatter(
                x=[trade.name],  # assume index aligns with price_series
                y=[trade["entry_price"]],
                mode="markers",
                marker={"symbol": marker_symbol, "color": marker_color, "size": 10},
                name=f"{pos.capitalize()} Entry",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[trade.name],
                y=[trade["exit_price"]],
                mode="markers",
                marker={"symbol": marker_symbol, "color": marker_color, "size": 10, "line": {"width": 1}},
                name=f"{pos.capitalize()} Exit",
            )
        )

    fig.update_layout(title=title or "Price & Trades", xaxis_title="Step", yaxis_title="Price")
    return fig


def equity_curve_figure(equity_curve: pd.Series, title: str | None = None) -> go.Figure:
    """Build an equity curve line chart."""
    fig = go.Figure(
        data=[
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode="lines",
                name="Equity Curve",
                line={"color": "#ff7f0e"},
            )
        ]
    )
    fig.update_layout(title=title or "Equity Curve", xaxis_title="Step", yaxis_title="Equity")
    return fig
