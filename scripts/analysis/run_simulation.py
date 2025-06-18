"""Streamlit UI for interactive back-testing of RL trading agents.

Usage:
    uv run streamlit run scripts/analysis/run_simulation.py
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import mlflow
import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from scripts.analysis.backend import run_simulation
from src.simulation.visualisation import equity_curve_figure, price_with_trades, tradingview_line_with_markers
from src.utils.logger import get_logger
from scripts.analysis.helpers import fetch_models_with_timeout
from src.utils.mlflow_utils import latest_version

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config - could be moved to YAML later
# ---------------------------------------------------------------------------
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
# Optional env fallbacks for UI defaults
DEFAULT_SYMBOL = os.getenv("DEFAULT_SYMBOL", "EUR/USD")
DEFAULT_TIMEFRAME = os.getenv("DEFAULT_TIMEFRAME", "1h")
FEATURES_DIR = Path(os.getenv("FEATURES_DIR", "data/processed/features"))

# Number of initial bars to display before first replay step (can be overwritten by model config)
DEFAULT_WARMUP_BARS = int(os.getenv("WARMUP_BARS", "50"))

# ---------------------------------------------------------------------------
# Helper: build lightweight-charts candlestick config with markers
# ---------------------------------------------------------------------------

def tradingview_candles_with_markers(price_df: pd.DataFrame, trades_df: pd.DataFrame):
    """Return (charts, key) tuple for candlestick view using lightweight-charts."""
    import hashlib

    if not {"open", "high", "low", "close"}.issubset(price_df.columns):
        # Fallback to line helper
        return tradingview_line_with_markers(price_df["close"], trades_df)

    # time column (yyyy-mm-dd)
    if isinstance(price_df.index, pd.DatetimeIndex):
        times = price_df.index.strftime("%Y-%m-%d")
    else:
        base = pd.Timestamp.utcnow().normalize()
        times = [(base + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in price_df.index]

    candle_data = [
        {
            "time": t,
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
        }
        for t, o, h, l, c in zip(
            times, price_df["open"], price_df["high"], price_df["low"], price_df["close"]
        )
    ]

    # markers similar to line helper
    marker_list = []
    for idx, row in trades_df.iterrows():
        if idx >= len(candle_data):
            continue
        side = str(row.get("position", "BUY")).upper()
        marker_list.append(
            {
                "time": candle_data[idx]["time"],
                "position": "belowBar" if side in {"BUY", "LONG"} else "aboveBar",
                "color": "rgba(38,166,154,1)" if side in {"BUY", "LONG"} else "rgba(239,83,80,1)",
                "shape": "arrowUp" if side in {"BUY", "LONG"} else "arrowDown",
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
                "type": "Candlestick",
                "data": candle_data,
                "markers": marker_list,
            }
        ],
    }

    key = "candles_" + hashlib.md5(str(len(candle_data)).encode()).hexdigest()
    return [chart_config], key

st.set_page_config(page_title="Interactive Simulation", layout="wide")

# ---------------------------------------------------------------------------
# Custom CSS to hide the default Streamlit "Running… / Stop" status indicator
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Hide the top-right running status indicator to reduce flicker during replay */
    div[data-testid="stStatusContainer"] {visibility: hidden !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar - model & period selection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔍 Simulation Controls")

    tracking_uri = st.text_input("MLflow Tracking URI", value=DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)

    # Fetch models from registry with short timeout to avoid UI freeze
    with st.spinner("Connecting to MLflow …"):
        client = MlflowClient()
        models = fetch_models_with_timeout(client, timeout=3)

    model_names = [m.name for m in models]
    models_present = bool(model_names)
    if not models_present:
        st.warning(
            "No models found in the MLflow Model Registry.  Guidance will appear in the main pane."
        )

    model_name = None  # initialise
    if models_present:
        model_name = st.selectbox("Choose a Model", options=model_names)
        try:
            latest_ver = latest_version(model_name)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        version_labels = [latest_ver]
        selected_label = st.selectbox("Version", options=version_labels, index=0)
        model_version = selected_label

    start_date = st.date_input(
        "Start Date", value=datetime.now() - timedelta(days=180), key="start_date"
    )
    end_date = st.date_input("End Date", value=datetime.now(), key="end_date")

    symbol = st.text_input("Symbol", value=DEFAULT_SYMBOL)
    timeframe = st.text_input("Timeframe", value=DEFAULT_TIMEFRAME)

    initial_balance = st.number_input("Initial Balance", min_value=1000.0, value=10_000.0, step=1000.0)

    run_button = st.button("Run Simulation 🚀")

# If page reruns (e.g., during replay) and simulation already happened,
# repopulate variables so we can display without running simulation again.
if not run_button and "sim_result" in st.session_state:
    result = st.session_state.sim_result
    session_path = st.session_state.sim_session_path
    price_series = st.session_state.price_series
    cached_display = True
else:
    cached_display = False

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

DEBUG_UI = os.getenv("DEBUG_STREAMLIT", "0") == "1"

if run_button:
    model_uri = f"models:/{model_name}/{model_version}"

    # Use a placeholder so we never display more than one status banner
    status = st.empty()
    status.info("Running simulation – this may take a few seconds…", icon="⏳")

    # Determine data path from user-provided values
    sanitized_symbol = symbol.replace("/", "")
    data_path = FEATURES_DIR / f"{sanitized_symbol}_{timeframe}_features.csv"

    try:
        result, session_path = run_simulation(
            model_uri=str(model_uri),
            data_path=str(data_path),
            initial_balance=initial_balance,
        )
    except FileNotFoundError as e:
        status.error(str(e), icon="❌")
        st.stop()
    except Exception as exc:
        logger.exception("Simulation failed: %s", exc)
        status.error(f"Simulation failed: {exc}", icon="❌")
        st.stop()

    # Persist into session_state for replay across reruns
    st.session_state.sim_result = result
    st.session_state.sim_session_path = session_path

    # Store price series for later display
    df = pd.read_csv(data_path)
    st.session_state.price_series = df["close"]
    st.session_state.price_df = df

    status.success("Simulation finished", icon="✅")

    # Immediately rerun so that UI is built in the cached_display branch only
    st.rerun()

    # --- The code below will not be reached because of st.rerun() ---
    # Tabs for visualisation
    if DEBUG_UI:
        st.caption("[debug] Simulation finished; building tabs …")

    base_tabs = ["📈 Price & Trades", "💰 Equity Curve", "📊 Metrics"]
    if session_path is not None:
        tabs = ["▶️ Replay"] + base_tabs
    else:
        tabs = base_tabs

    tab_objs = st.tabs(tabs)
    tab_map = {name: obj for name, obj in zip(tabs, tab_objs)}

    price_tab = tab_map["📈 Price & Trades"]
    equity_tab = tab_map["💰 Equity Curve"]
    metrics_tab = tab_map["📊 Metrics"]
    replay_tab = tab_map.get("▶️ Replay")

    with price_tab:
        st.subheader("Price & Trades (Plotly)")
        st.plotly_chart(price_with_trades(price_series, result.trades), use_container_width=True, key="price_plot")

        # Try TradingView chart if component available
        try:
            from streamlit_lightweight_charts import renderLightweightCharts

            charts, tv_key = tradingview_line_with_markers(price_series, result.trades)
            st.subheader("TradingView Chart (beta)")
            renderLightweightCharts(charts, "tv_price_chart")
        except ModuleNotFoundError:
            st.info("Install 'streamlit-lightweight-charts' to view the TradingView chart.")

    with equity_tab:
        st.subheader("Equity Curve")
        st.plotly_chart(equity_curve_figure(result.equity_curve), use_container_width=True)

    with metrics_tab:
        st.subheader("Performance Metrics")
        st.json(result.metrics)
        st.dataframe(result.trades)

    if replay_tab is not None:
        with replay_tab:
            st.subheader("Interactive Replay")

            import time as _time

            # Load session once into session_state
            if "replay_df" not in st.session_state or st.session_state.get("replay_loaded_path") != str(session_path):
                df_replay = pd.read_parquet(session_path)
                st.session_state.replay_df = df_replay
                # reuse price series
                st.session_state.price_series = price_series
                st.session_state.trades_df = result.trades
                st.session_state.warmup_bars = DEFAULT_WARMUP_BARS
                st.session_state.replay_idx = st.session_state.warmup_bars
                st.session_state.playing = False
                st.session_state.speed = 1
                st.session_state.replay_loaded_path = str(session_path)

            df_replay = st.session_state.replay_df
            warmup_bars = st.session_state.get("warmup_bars", DEFAULT_WARMUP_BARS)
            max_step = len(df_replay) - 1

            # Controls
            cols = st.columns([1,3,6,2])
            with cols[0]:
                st.markdown("<div style='height: 100%; display: flex; align-items: center;'>", unsafe_allow_html=True)
                if st.button("▶️ Play" if not st.session_state.playing else "⏸ Pause", key="play_button"):
                    st.session_state.playing = not st.session_state.playing
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            st.session_state.speed = cols[1].slider("Speed (steps/s)", 1, 10, st.session_state.speed, key="speed_slider")

            st.session_state.replay_idx = cols[2].slider(
                "Position", 0, max_step, st.session_state.replay_idx, key="pos_slider", disabled=st.session_state.playing
            )

            chart_styles = ["Line", "Candles"]
            current_style = st.session_state.get("chart_style", "Line")
            selected_style = cols[3].selectbox("Chart", chart_styles, index=chart_styles.index(current_style), key="chart_style_select")
            if selected_style != current_style:
                st.session_state.chart_style = selected_style
                st.rerun()

            # Render chart for current index
            idx = st.session_state.replay_idx
            price_subset_df = st.session_state.price_df.iloc[: idx + 1]
            price_subset_series = price_subset_df["close"]
            trades_subset = st.session_state.trades_df.iloc[: idx + 1]

            style = st.session_state.get("chart_style", "Line")
            if style == "Line":
                charts, _ = tradingview_line_with_markers(price_subset_series, trades_subset)
            else:
                charts, _ = tradingview_candles_with_markers(price_subset_df, trades_subset)
            renderLightweightCharts(charts, "tv_replay_chart")

            # Auto-advance
            if st.session_state.playing:
                if idx < max_step:
                    st.session_state.replay_idx += 1
                    _time.sleep(1.0 / st.session_state.speed)
                    st.rerun()
                else:
                    st.session_state.playing = False

# ----------------------------------------------------
# Display cached simulation on rerun
# ----------------------------------------------------
if cached_display:
    # Ensure necessary session_state items are present after initial rerun
    if "price_series" not in st.session_state:
        st.error("Internal error: price_series missing from session_state.")
        st.stop()

    price_series = st.session_state.price_series
    # Ensure price_df present
    if "price_df" not in st.session_state:
        df_tmp = pd.read_csv(st.session_state.sim_session_path.replace("sessions", "../processed/features").replace(".parquet", "_features.csv")) if False else None
        # Fallback: rebuild from price_series only
        st.session_state.price_df = pd.DataFrame({"close": price_series})

    if "replay_df" not in st.session_state:
        try:
            df_replay = pd.read_parquet(st.session_state.sim_session_path)
        except Exception as _load_exc:
            st.error(f"Failed to load replay session data: {_load_exc}")
            st.stop()

        st.session_state.replay_df = df_replay
        st.session_state.trades_df = st.session_state.sim_result.trades
        st.session_state.warmup_bars = DEFAULT_WARMUP_BARS
        st.session_state.replay_idx = st.session_state.warmup_bars
        st.session_state.playing = False
        st.session_state.speed = 1

    # Build tabs identical to above but reuse result & price_series
    base_tabs = ["📈 Price & Trades", "💰 Equity Curve", "📊 Metrics"]
    tabs = ["▶️ Replay"] + base_tabs
    tab_objs = st.tabs(tabs)
    tab_map = {name: obj for name, obj in zip(tabs, tab_objs)}

    price_tab = tab_map["📈 Price & Trades"]
    equity_tab = tab_map["💰 Equity Curve"]
    metrics_tab = tab_map["📊 Metrics"]
    replay_tab = tab_map.get("▶️ Replay")

    with price_tab:
        st.subheader("Price & Trades (Plotly)")
        st.plotly_chart(price_with_trades(price_series, result.trades), use_container_width=True, key="price_plot")

        try:
            from streamlit_lightweight_charts import renderLightweightCharts
            charts, tv_key = tradingview_line_with_markers(price_series, result.trades)
            st.subheader("TradingView Chart (beta)")
            renderLightweightCharts(charts, "tv_price_chart")
        except ModuleNotFoundError:
            st.info("Install 'streamlit-lightweight-charts' to view the TradingView chart.")

    with equity_tab:
        st.subheader("Equity Curve")
        st.plotly_chart(equity_curve_figure(result.equity_curve), use_container_width=True)

    with metrics_tab:
        st.subheader("Performance Metrics")
        st.json(result.metrics)
        st.dataframe(result.trades)

    # Replay tab uses same logic as before but without price_series re-import.
    with replay_tab:
        st.subheader("Interactive Replay")
        import time as _time

        df_replay = st.session_state.replay_df
        warmup_bars = st.session_state.get("warmup_bars", DEFAULT_WARMUP_BARS)
        max_step = len(df_replay) - 1

        cols = st.columns([1,3,6,2])
        with cols[0]:
            st.markdown("<div style='height: 100%; display: flex; align-items: center;'>", unsafe_allow_html=True)
            if st.button("▶️ Play" if not st.session_state.playing else "⏸ Pause", key="play_button_cached"):
                st.session_state.playing = not st.session_state.playing
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.session_state.speed = cols[1].slider("Speed (steps/s)", 1, 10, st.session_state.speed, key="speed_slider")

        st.session_state.replay_idx = cols[2].slider(
            "Position", 0, max_step, st.session_state.replay_idx, key="pos_slider", disabled=st.session_state.playing
        )

        chart_styles = ["Line", "Candles"]
        current_style = st.session_state.get("chart_style", "Line")
        selected_style = cols[3].selectbox("Chart", chart_styles, index=chart_styles.index(current_style), key="chart_style_select_cached")
        if selected_style != current_style:
            st.session_state.chart_style = selected_style
            st.rerun()

        idx = st.session_state.replay_idx
        price_subset_df = st.session_state.price_df.iloc[: idx + 1]
        price_subset_series = price_subset_df["close"]
        trades_subset = result.trades.iloc[: idx + 1]

        style = st.session_state.get("chart_style", "Line")
        if style == "Line":
            charts, _ = tradingview_line_with_markers(price_subset_series, trades_subset)
        else:
            charts, _ = tradingview_candles_with_markers(price_subset_df, trades_subset)
        renderLightweightCharts(charts, "tv_replay_chart")

        if st.session_state.playing:
            if idx < max_step:
                st.session_state.replay_idx += 1
                _time.sleep(1.0 / st.session_state.speed)
                st.rerun()
            else:
                st.session_state.playing = False

# --- Early exit if no models ---
if not models_present:
    st.markdown(
        "## 📂 No Models Found\n"
        "The MLflow Model Registry is empty.\n\n"
        "**Next steps:**\n"
        "1. Generate historical data with `download_historical.py`.\n"
        "2. Convert to Qlib bin files via `dump_bin.py`.\n"
        "3. Build features with `build_features.py`.\n"
        "4. Train an agent with `train_agent.py` (this logs the model).\n\n"
        "Return here and click **Rerun** (or press `r`)."
    )