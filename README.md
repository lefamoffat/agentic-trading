# Agentic Trading - RL Trading System

A sophisticated multi-asset reinforcement learning trading system with a broker-agnostic architecture, built with Microsoft Qlib and PyTorch.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This project implements a reinforcement learning-based trading system with a modular, broker-agnostic architecture supporting multiple asset classes. It leverages a powerful combination of institutional-grade quantitative tools and cutting-edge machine learning libraries to create a robust, end-to-end solution for algorithmic trading.

The system features a **fully dynamic, model-agnostic environment** that can adapt to any ML framework and trading configuration through YAML-based configuration.

For a deeper dive into the system's design, please see the [**Architecture Philosophy**](docs/architecture.md) and the [**Implementation Plan**](docs/implementation_plan.md).

## 🚀 Features

-   ✅ **Dynamic Model-Agnostic Environment**: Accepts actions from any ML framework (SB3, PyTorch, LLMs, human input)
-   ✅ **Configurable Observations**: Dynamic feature selection, time-aware trading, portfolio state tracking
-   ✅ **Broker-Agnostic Architecture**: Modular design supporting multiple brokers
-   ✅ **Qlib-Powered Feature Engineering**: Leverages Qlib's alpha libraries for advanced feature creation
-   ✅ **Standardized Data Pipeline**: Consistent data processing across all sources
-   ✅ **Live Trading Integration**: Production-ready broker integrations
-   ✅ **Comprehensive Backtesting**: Rigorous performance analysis via Qlib
-   ✅ **Experiment Tracking**: Integrated with MLflow for logging runs, metrics, and models
-   ✅ **Hyperparameter Optimization**: Built-in Optuna script for automated HPO
-   ✅ **Comprehensive Test Coverage**: 134/134 tests passing with robust validation

## 📚 Documentation

| Document                                               | Description                                                                |
| ------------------------------------------------------ | -------------------------------------------------------------------------- |
| [**Getting Started**](docs/getting-started.md)         | A step-by-step guide to install, configure, and run the project.           |
| [**Dynamic Environment**](docs/dynamic_environment.md) | Guide to the model-agnostic, configurable trading environment system.      |
| [**Configuration**](docs/configuration.md)             | Detailed reference for all configuration files and environment variables.  |
| [**Technical Overview**](docs/technical-overview.md)   | An overview of the project structure, feature engineering, and monitoring. |
| [**Development Guide**](docs/development.md)           | Guidelines for testing, code quality, and contributing to the project.     |
| [**Architecture Philosophy**](docs/architecture.md)    | Core principles guiding the project's design and library usage.            |
| [**Implementation Plan**](docs/implementation_plan.md) | The phased roadmap for the project's evolution.                            |

## 📋 Prerequisites

-   Python 3.11+
-   [uv](https://github.com/astral-sh/uv) package manager
-   API keys for e.g. data providers, broker integrations (e.g. forex.com), etc.

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

The system uses YAML configuration files in the `configs/` directory:

-   **`agent_config.yaml`** - RL hyperparameters and training settings
-   **`data_config.yaml`** - Data sources and API configurations
-   **`trading_config.yaml`** - Trading parameters and risk management
-   **`qlib_config.yaml`** - Qlib initialization and setup

## 📊 Project Structure

```
agentic-trading/
├── configs/                 # YAML configuration files
├── src/                     # Main source code
│   ├── environment/         # Dynamic, model-agnostic RL environment
│   ├── intelligence/        # LLM-based configuration intelligence
│   ├── agents/              # RL agent implementations (Stable-Baselines3)
│   ├── brokers/             # Broker integrations (forex.com working)
│   ├── market_data/         # Centralized market data handling
│   ├── data/                # Data utilities & market calendars
│   ├── strategies/          # Trading strategy implementations
│   ├── callbacks/           # Training callbacks
│   ├── utils/               # Core utilities
│   └── types/               # Centralized type definitions
├── scripts/                 # Executable scripts
│   ├── features/            # Qlib-based feature generation
│   ├── data/                # Data preparation scripts
│   ├── training/            # RL agent training scripts
│   └── setup/               # Project initialization
├── data/                    # Data storage (raw, processed, qlib, models)
├── integration_tests/       # Integration tests
└── docs/                    # Documentation
```

## 🔄 Dynamic Environment System

The trading environment is **fully dynamic and model-agnostic**:

### Model-Agnostic Actions

```python
# Works with ANY ML framework:
env.step(np.array([1]))          # Stable-Baselines3
env.step(torch.tensor([0]))      # PyTorch
env.step("buy")                  # LLM string commands
env.step(2)                      # Human input
```

### Dynamic Observations

```python
# Configure any features via YAML:
observation_features: ['close', 'volume', 'rsi', 'custom_indicator']
include_time_features: true      # Market hours, day of week
include_portfolio_state: true    # Balance, positions, PnL
```

See [**Dynamic Environment Guide**](docs/dynamic_environment.md) for complete details.

## 🎮 Quick Start

### 1. Launch MLflow Server (Optional, for Tracking)

To track experiments, parameters, and metrics, first launch the MLflow server.

```bash
uv run scripts/setup/launch_mlflow.sh
```

Access the UI at [http://localhost:5001](http://localhost:5001).

### 2. Prepare Training Data

Prepare market data using the centralized market_data module. This replaces the old multi-step process with a single command:

```bash
# Prepare data using the new market_data module
uv run scripts/data/prepare_data.py --symbol "EUR/USD" --timeframe 1h --days 365
```

### 3. Train an RL Agent

Now you can run the training pipeline. This will automatically prepare data using the market_data module and log the run to MLflow if the server is active.

```bash
# Start a new training run (includes automatic data preparation)
uv run scripts/training/train_agent.py --symbol "EUR/USD" --timeframe 1h --timesteps 20000 --days 365
```

### 4. Optimize Hyperparameters

To find the best hyperparameters for an agent, use the optimization script. This will run multiple training trials and log them as nested runs in MLflow.

```bash
uv run scripts/training/optimize_agent.py --symbol "EUR/USD" --timeframe 1h --timesteps 5000 --trials 20
```

**Note:** The optimization script will be updated in a future version to use the new market_data module.

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

```

```
