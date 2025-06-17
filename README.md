# Agentic Trading - EUR/USD RL Trading System

A sophisticated multi-asset reinforcement learning trading system with a broker-agnostic architecture, built with Microsoft Qlib and PyTorch.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This project implements a reinforcement learning-based trading system with a modular, broker-agnostic architecture supporting multiple asset classes. It leverages a powerful combination of institutional-grade quantitative tools and cutting-edge machine learning libraries to create a robust, end-to-end solution for algorithmic trading.

For a deeper dive into the system's design, please see the [**Architecture Philosophy**](docs/architecture.md) and the [**Implementation Plan**](docs/implementation_plan.md).

## 🚀 Features

-   ✅ **Broker-Agnostic Architecture**: Modular design supporting multiple brokers.
-   ✅ **Qlib-Powered Feature Engineering**: Leverages Qlib's alpha libraries for advanced feature creation.
-   ✅ **Standardized Data Pipeline**: Consistent data processing across all sources.
-   ✅ **Gymnasium-based RL Environments**: Standardized environments for agent training.
-   ✅ **Live Trading Integration**: Production-ready broker integrations.
-   ✅ **Comprehensive Backtesting**: Rigorous performance analysis via Qlib.
-   ✅ **Experiment Tracking**: Integrated with MLflow for logging runs, metrics, and models.
-   ✅ **Hyperparameter Optimization**: Built-in Optuna script for automated HPO.
-   ✅ **100% Test Coverage**: Comprehensive validation for all core components.

## 📚 Documentation

| Document                                               | Description                                                                |
| ------------------------------------------------------ | -------------------------------------------------------------------------- |
| [**Getting Started**](docs/getting-started.md)         | A step-by-step guide to install, configure, and run the project.           |
| [**Configuration**](docs/configuration.md)             | Detailed reference for all configuration files and environment variables.  |
| [**Technical Overview**](docs/technical-overview.md)   | An overview of the project structure, feature engineering, and monitoring. |
| [**Development Guide**](docs/development.md)           | Guidelines for testing, code quality, and contributing to the project.     |
| [**Architecture Philosophy**](docs/architecture.md)    | Core principles guiding the project's design and library usage.            |
| [**Implementation Plan**](docs/implementation_plan.md) | The phased roadmap for the project's evolution.                            |

## 📋 Prerequisites

-   Python 3.11+
-   [uv](https://github.com/astral-sh/uv) package manager
-   API keys for data providers (forex.com)

## 🛠️ Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/lefamoffat/agentic-trading.git
    cd agentic-trading
    ```

2. **Install dependencies**

    ```bash
    uv sync
    ```

3. **Initialize the project**

    ```bash
    uv run scripts/setup/init_project.py
    ```

4. **Set up environment variables**
    ```bash
    cp .env.example .env
    # Edit .env with your API keys
    ```

## 🔧 Configuration

The system uses YAML configuration files in the `config/` directory:

-   **`agent_config.yaml`** - RL hyperparameters and training settings
-   **`data_config.yaml`** - Data sources and API configurations
-   **`trading_config.yaml`** - Trading parameters and risk management
-   **`qlib_config.yaml`** - Qlib initialization and setup

### Environment Variables (.env)

```bash
# API Keys
FOREX_COM_USERNAME=your_username_here
FOREX_COM_PASSWORD=your_password_here
FOREX_COM_APP_KEY=your_app_key_here
FOREX_COM_SANDBOX=true

# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO

# Trading
POSITION_SIZE=10000
MAX_DRAWDOWN=0.10
STOP_LOSS=0.02
```

## 📊 Project Structure

```
agentic-trading/
├── config/                 # YAML configuration files
├── src/                    # Main source code
│   ├── models/            # RL agent implementations (Stable-Baselines3)
│   ├── brokers/           # Broker integrations (forex.com working)
│   ├── data/              # Data handling & market calendars
│   ├── environments/      # Gymnasium-based RL environments
│   ├── strategies/        # Trading strategy implementations
│   ├── utils/             # Core utilities
│   ├── types/             # Centralized type definitions
│   └── exceptions.py      # Custom exception hierarchy
├── scripts/               # Executable scripts
│   ├── features/          # Qlib-based feature generation
│   ├── training/          # RL agent training scripts
│   └── setup/             # Project initialization
├── data/                  # Data storage (raw, processed, qlib, models)
├── logs/                  # Log files (system and tensorboard)
├── results/               # Results and reports
└── tests/                 # Unit and integration tests
```

## 🎮 Quick Start

### 1. Launch MLflow Server (Optional, for Tracking)

To track experiments, parameters, and metrics, first launch the MLflow server.

```bash
uv run scripts/setup/launch_mlflow.sh
```

Access the UI at [http://localhost:5001](http://localhost:5001).

### 2. Build Features with Qlib

Before training, you must generate features from the raw data using Qlib.

```bash
uv run scripts/features/build_features.py --symbol "EUR/USD" --timeframe 1h
```

### 3. Train an RL Agent

Now you can run the training pipeline. This will use the features generated in the previous step and log the run to MLflow if the server is active.

```bash
# Start a new training run
uv run scripts/training/train_agent.py --symbol "EUR/USD" --timeframe 1h --timesteps 20000
```

### 4. Optimize Hyperparameters

To find the best hyperparameters for an agent, use the optimization script. This will run multiple training trials and log them as nested runs in MLflow.

```bash
uv run scripts/training/optimize_agent.py --symbol "EUR/USD" --timeframe 1h --timesteps 5000 --trials 20
```

### 5. Run Tests

```bash
# Run unit tests (default, fast)
uv run scripts/run_tests.py

# Run all tests (unit and integration)
uv run scripts/run_tests.py --all

# Run only integration tests (requires credentials)
uv run scripts/run_tests.py --integration
```

### 6. Test Broker Integration

```bash
# Test forex.com broker integration (requires credentials)
uv run scripts/run_tests.py --integration -k "test_real_get_live_price"

# Test broker authentication
uv run scripts/run_tests.py --integration -k "test_real_authentication_is_successful"
```

## 🧠 Technical Indicators Available

Feature engineering is handled by **Microsoft Qlib**, which provides access to a vast library of technical indicators and alpha factors, including the renowned `Alpha158` and `Alpha360` collections. The feature generation process is configurable and extensible within the `scripts/features/build_features.py` script.

## 🔍 Monitoring & Analysis

### Logging

-   Structured logging with loguru
-   Separate logs for training, backtesting, and trading
-   Configurable log levels

### Data Quality Metrics

-   Completeness scoring
-   Gap detection and analysis
-   Outlier identification
-   Consistency validation

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk and may not be suitable for all investors. Past performance is not indicative of future results.
