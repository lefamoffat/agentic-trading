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
from src.simulation.visualisation import equity_curve_figure, price_with_trades
from src.utils.logger import get_logger
from scripts.analysis.helpers import fetch_models_with_timeout

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config - could be moved to YAML later
# ---------------------------------------------------------------------------
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
# Optional env fallbacks for UI defaults
DEFAULT_SYMBOL = os.getenv("DEFAULT_SYMBOL", "EUR/USD")
DEFAULT_TIMEFRAME = os.getenv("DEFAULT_TIMEFRAME", "1h")
FEATURES_DIR = Path(os.getenv("FEATURES_DIR", "data/processed/features"))

st.set_page_config(page_title="Interactive Simulation", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar - model & period selection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔍 Simulation Controls")

    tracking_uri = st.text_input("MLflow Tracking URI", value=DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)

    # Fetch models from registry with short timeout to avoid UI freeze
    client = MlflowClient()
    with st.spinner("Connecting to MLflow …"):
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
        # Fetch versions for selected model
        versions = client.get_latest_versions(model_name)
        version_labels = [f"{v.version}" for v in versions]
        selected_label = st.selectbox("Version", options=version_labels, index=0)
        model_version = versions[version_labels.index(selected_label)].version

    start_date = st.date_input(
        "Start Date", value=datetime.now() - timedelta(days=180), key="start_date"
    )
    end_date = st.date_input("End Date", value=datetime.now(), key="end_date")

    symbol = st.text_input("Symbol", value=DEFAULT_SYMBOL)
    timeframe = st.text_input("Timeframe", value=DEFAULT_TIMEFRAME)

    initial_balance = st.number_input("Initial Balance", min_value=1000.0, value=10_000.0, step=1000.0)

    run_button = st.button("Run Simulation 🚀")

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
        result = run_simulation(
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

    status.success("Simulation finished", icon="✅")

    # Tabs for visualisation
    if DEBUG_UI:
        st.caption("[debug] Simulation finished; building tabs …")
    tab1, tab2, tab3 = st.tabs(["📈 Price & Trades", "💰 Equity Curve", "📊 Metrics"])

    # Need price series for first tab
    df = pd.read_csv(data_path)
    price_series = df["close"]

    with tab1:
        st.subheader("Price & Trades")
        st.plotly_chart(price_with_trades(price_series, result.trades), use_container_width=True)

    with tab2:
        st.subheader("Equity Curve")
        st.plotly_chart(equity_curve_figure(result.equity_curve), use_container_width=True)

    with tab3:
        st.subheader("Performance Metrics")
        st.json(result.metrics)
        st.dataframe(result.trades)

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
    st.stop()

# --- Debug output -------------------------------------------------
logger.info("Streamlit UI: %s models fetched from registry", len(models))
if DEBUG_UI:
    st.caption(f"[debug] models fetched: {len(models)}")
