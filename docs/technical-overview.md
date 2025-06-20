# Technical Overview

This document provides a high-level overview of the key technical components of the Agentic Trading system.

## Project Structure

The codebase is organized into several key directories:

```
agentic-trading/
├── configs/                 # YAML configuration files
├── src/                    # Main source code
│   ├── models/            # RL agent implementations
│   ├── brokers/           # Broker integrations
│   ├── data/              # Data handling & market calendars
│   ├── environments/      # Gymnasium-based RL environments
│   └── utils/             # Core utilities
├── scripts/               # Executable scripts for training, features, etc.
├── data/                  # Data storage (raw, processed, qlib, models)
├── logs/                  # Log files
└── tests/                 # Unit and integration tests
```

For a more detailed breakdown, refer to the [Architecture Guide](architecture.md).

## Feature Engineering

Feature engineering is handled exclusively by **Microsoft Qlib**. This library provides access to a vast collection of technical indicators and alpha factors, including the renowned `Alpha158` and `Alpha360` collections.

The primary script for this process is `scripts/features/build_features.py`. It downloads the latest market data and uses Qlib to generate a rich feature set, which is then stored and used for training the RL agent. The process is configurable and can be extended to include custom features.

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
