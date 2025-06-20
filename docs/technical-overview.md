# Technical Overview

This document provides a high-level overview of the key technical components of the Agentic Trading system.

## Project Structure

The codebase is organized into several key directories:

```
agentic-trading/
├── configs/                 # YAML configuration files
├── src/                    # Main source code
│   ├── agents/            # RL agent implementations (Stable-Baselines3)
│   ├── brokers/           # Broker integrations (forex.com working)
│   ├── market_data/       # Centralized market data handling
│   ├── data/              # Data utilities & market calendars
│   ├── environment/       # Dynamic, model-agnostic RL environment
│   ├── intelligence/      # LLM-based configuration intelligence
│   ├── callbacks/         # Training callbacks
│   ├── utils/             # Core utilities
│   └── types/             # Centralized type definitions
├── scripts/               # Executable scripts
│   ├── features/          # Qlib-based feature generation
│   ├── data/              # Data preparation scripts
│   ├── training/          # RL agent training scripts
│   └── setup/             # Project initialization
├── data/                  # Data storage (raw, processed, qlib, models)
├── integration_tests/     # Integration tests
└── docs/                  # Documentation
```

For a more detailed breakdown, refer to the [Architecture Guide](architecture.md).

## Market Data & Feature Engineering

### Centralized Market Data Module

The system uses a centralized `src/market_data/` module that provides:

-   **Source-agnostic data fetching** from brokers, APIs, and other providers
-   **Intelligent caching** to avoid re-downloading data
-   **Qlib binary format integration** for efficient feature engineering
-   **Type-safe data contracts** using Pydantic models
-   **Date range based data retrieval** inspired by Alpaca's API

### Feature Engineering with Qlib

Feature engineering is handled exclusively by **Microsoft Qlib**, providing access to a vast collection of technical indicators and alpha factors, including the renowned `Alpha158` and `Alpha360` collections.

The system integrates Qlib through the market_data module, which automatically handles:

-   Data format conversion to Qlib's binary format
-   Feature calculation using Qlib's alpha libraries
-   Seamless integration with the training pipeline

### Simplified Data Preparation

The old multi-step process has been replaced with a single command:

```bash
# Old approach (deprecated)
scripts/data/download_historical.py → scripts/data/dump_bin.py → scripts/features/build_features.py

# New approach (current)
scripts/data/prepare_data.py  # Uses market_data module internally
```

## Monitoring & Analysis

### Logging

-   **Structured Logging**: We use `loguru` for structured and configurable logging.
-   **Log Separation**: The system generates separate logs for different processes like training, backtesting, and live trading, which are stored in the `logs/` directory.

### Experiment Tracking with MLflow

-   **MLflow Integration**: All training and optimization runs are logged as experiments in MLflow.
-   **Tracked Information**: This includes hyperparameters, performance metrics (e.g., Sharpe Ratio, Profit), and the saved model artifacts.
-   **UI**: An MLflow tracking server can be launched with `bash scripts/setup/launch_mlflow.sh` to provide a web interface for comparing runs.

### Data Quality

The data pipeline includes checks to ensure the quality and integrity of market data:

-   Completeness scoring
-   Gap detection and analysis
-   Outlier identification
-   Consistency validation
