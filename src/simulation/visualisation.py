"""Plotly figure builders for simulation results."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from typing import Any

__all__ = [
    "price_with_trades",
    "equity_curve_figure",
    "tradingview_line_with_markers",
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


def tradingview_line_with_markers(price_series: pd.Series, trades: list[dict[str, Any]]):
    """Return a (charts, key) tuple for *streamlit-lightweight-charts*.

    Args:
        price_series: pd.Series indexed by integer or datetime; must have name "close" or similar.
        trades: List of trade dicts with at least ``index`` (int) and ``side`` ("BUY"/"SELL").

    Returns:
        charts (list[dict]): ready for ``renderLightweightCharts`` and a stable unique key.
    """
    import json
    import hashlib

    # Build main line series data {time, value}
    if isinstance(price_series.index, pd.DatetimeIndex):
        times = price_series.index.strftime("%Y-%m-%d")
    else:
        # Assume integer index representing steps → build synthetic dates
        base = pd.Timestamp.utcnow().normalize()
        times = [(base + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in price_series.index]

    line_data = [{"time": t, "value": float(v)} for t, v in zip(times, price_series.values)]

    # Build markers for trades
    marker_list: list[dict[str, Any]] = []
    if isinstance(trades, pd.DataFrame):
        iterable = trades.iterrows()
        for idx, row in iterable:  # idx aligns with bar index in Backtester
            side_raw = str(row.get("position", "BUY")).upper()
            side = "BUY" if side_raw in {"BUY", "LONG", "BULL"} else "SELL"
            if idx >= len(line_data):
                continue
            marker_list.append(
                {
                    "time": line_data[idx]["time"],
                    "position": "belowBar" if side == "BUY" else "aboveBar",
                    "color": "rgba(38,166,154,1)" if side == "BUY" else "rgba(239,83,80,1)",
                    "shape": "arrowUp" if side == "BUY" else "arrowDown",
                    "text": side,
                }
            )
    else:
        for trade in trades:  # type: ignore[assignment]
            if not isinstance(trade, dict):
                continue
            idx = trade.get("index")
            if idx is None or idx >= len(line_data):
                continue
            side = trade.get("side", "BUY").upper()
            marker_list.append(
                {
                    "time": line_data[idx]["time"],
                    "position": "belowBar" if side == "BUY" else "aboveBar",
                    "color": "rgba(38,166,154,1)" if side == "BUY" else "rgba(239,83,80,1)",
                    "shape": "arrowUp" if side == "BUY" else "arrowDown",
                    "text": side,
                }
            )

    chart_config = {
        "chart": {
            "height": 400,
            "layout": {
                "background": {"type": "solid", "color": "#131722"},
                "textColor": "#d1d4dc",
            },
            "grid": {
                "vertLines": {"color": "rgba(42, 46, 57, 0.4)"},
                "horzLines": {"color": "rgba(42, 46, 57, 0.4)"},
            },
        },
        "series": [
            {
                "type": "Line",
                "data": line_data,
                "options": {"lineWidth": 2, "color": "#26a69a"},
                "markers": marker_list,
            }
        ],
    }

    # Key based on hash of data length for streamlit component uniqueness
    key = "line_" + hashlib.md5(str(len(line_data)).encode()).hexdigest()
    return [chart_config], key
