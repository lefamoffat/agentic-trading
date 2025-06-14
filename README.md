# Agentic Trading - EUR/USD RL Trading System

A sophisticated multi-asset reinforcement learning trading system with a broker-agnostic architecture, built with Microsoft Qlib and PyTorch.

## 🎯 Project Overview

This project implements a reinforcement learning-based trading system with a modular, broker-agnostic architecture supporting multiple asset classes. It leverages:

-   **Microsoft Qlib** for institutional-grade feature engineering and quantitative analysis.
-   **Stable-Baselines3** for robust, pre-built reinforcement learning algorithms.
-   **Broker-Agnostic Design** with factory patterns for easy broker addition.
-   **Standardized Data Pipeline** ensuring consistent format across all sources.
-   **Gymnasium-based RL Environments** for standardized agent training.
-   **Live Trading Integration** with brokers (forex.com, others planned).
-   **Comprehensive Backtesting** and performance analysis via Qlib.

## 🚀 Features

-   ✅ **Broker-Agnostic Architecture** - Modular design supporting multiple brokers (forex.com)
-   ✅ **Qlib-Powered Feature Engineering** - Leverages Qlib's alpha libraries for advanced feature creation.
-   ✅ **Standardized Data Pipeline** - Consistent CSV format across all data sources.
-   ✅ **Market Calendar System** - Modular calendar support (forex 24/5, stocks, crypto).
-   ✅ **RL Training Pipeline** - Automated training, evaluation, and model selection loop.
-   ✅ **Risk Management** - Built-in stop loss, take profit, and drawdown controls.
-   ✅ **Live Trading Ready** - Production-ready forex.com broker integration.
-   ✅ **Configurable** - YAML-based configuration for all parameters.
-   ✅ **Comprehensive Logging** - Detailed logging for training, trading, and TensorBoard visualization.
-   ✅ **100% Test Coverage** - Comprehensive validation for all core components.

## 📋 Prerequisites

-   Python 3.11+
-   [uv](https://github.com/astral-sh/uv) package manager
-   API keys for data providers (forex.com)

## 🛠️ Installation

1. **Clone the repository**

    ```bash
    git clone <repository-url>
    cd agentic-trading
    ```

2. **Install dependencies**

    ```bash
    uv sync
    ```

3. **Initialize the project**

    ```bash
    uv run python scripts/setup/init_project.py
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
│   ├── agents/            # RL agent implementations (PPO, etc.)
│   ├── brokers/           # Broker integrations (forex.com working)
│   ├── data/              # Data handling & market calendars
│   ├── environments/      # Gymnasium-based RL environments
│   ├── strategies/        # Trading strategy implementations
│   ├── utils/             # Core utilities
│   ├── types.py           # Centralized type definitions
│   └── exceptions.py      # Custom exception hierarchy
├── scripts/               # Executable scripts
│   ├── features/          # Qlib-based feature generation
│   │   └── build_features.py
│   ├── training/          # RL agent training scripts
│   │   └── train_agent.py
│   └── setup/             # Project initialization
├── data/                  # Data storage (raw, processed, qlib, models)
├── logs/                  # Log files (system and tensorboard)
├── results/               # Results and reports
└── tests/                 # Unit and integration tests
```

## 🎮 Quick Start

### 1. Build Features with Qlib

Before training, you must generate features from the raw data using Qlib.

```bash
uv run python scripts/features/build_features.py --symbol "EUR/USD" --timeframe 1h
```

### 2. Train an RL Agent

Now you can run the training pipeline. This will use the features generated in the previous step.

```bash
# Start a new training run
uv run python scripts/training/train_agent.py --symbol "EUR/USD" --timeframe 1h --timesteps 20000

# Resume a previous training run
uv run python scripts/training/train_agent.py --run-id <YOUR_RUN_ID>
```

### 3. Run Tests

```bash
# Run unit tests (default, fast)
uv run python scripts/run_tests.py

# Run all tests (unit and integration)
uv run python scripts/run_tests.py --all

# Run only integration tests (requires credentials)
uv run python scripts/run_tests.py --integration
```

### 4. Test Broker Integration

```bash
# Test forex.com broker integration (requires credentials)
uv run python scripts/run_tests.py --integration -k "test_real_get_live_price"

# Test broker authentication
uv run python scripts/run_tests.py --integration -k "test_real_authentication_is_successful"
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

## 🧪 Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Run specific test categories
uv run pytest tests/test_forex_com_broker.py -v
```

### Code Quality

```bash
# Type checking
uv run mypy src/

# Linting
uv run ruff check src/

# Formatting
uv run ruff format src/
```

## 📝 Development Status

**✅ Phase 1 Completed:**

-   Project structure and configuration
-   Core utilities (config loader, logging, settings)
-   YAML configuration files
-   Environment setup and project initialization

**✅ Phase 2 Completed:**

-   Broker abstraction layer with factory pattern
-   Modular market calendar system (forex, ready for stocks/crypto)
-   Data standardization pipeline with quality validation
-   Standardized CSV format: `timestamp,open,high,low,close,volume`

**✅ Phase 3 Completed (Now Deprecated):**

-   The original, custom feature engineering framework has been **deprecated and removed** in favor of a more robust Qlib-based pipeline.

**✅ Phase 4 Completed:**

-   Live forex.com broker integration (authentication, live prices, historical data)
-   Symbol mapping system supporting multiple brokers
-   Comprehensive test suite
-   DRY compliance and code refactoring
-   Production-ready error handling and logging

**✅ Phase 5 Completed:**

-   **Qlib Integration**: Successfully integrated Qlib as the core feature engineering engine.
-   **RL Environment**: Implemented a `gymnasium`-compatible trading environment.
-   **RL Agents**: Built a modular framework for RL agents using `stable-baselines3`.
-   **RL Training Pipeline**: Created a robust, end-to-end training script with automated evaluation, best-model saving, and TensorBoard logging.
-   **Strategy Framework**: Established a base for connecting agents to live trading execution.

**📋 Upcoming Phases:**

-   **Backtesting Framework**: Implement a Qlib-native backtesting strategy to evaluate trained agents.
-   **Hyperparameter Optimization**: Build an HPO pipeline using Optuna and Qlib to automate the discovery of optimal agent configurations.
-   **Live Trading Execution**: Refine the live trading strategy to be fully independent of the training environment.
-   **Risk Management and Portfolio Optimization**: Integrate advanced risk controls and portfolio management techniques.
-   **Add Test Coverage**: Add integration tests for the training pipeline and unit tests for the RL environment.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests (maintain 100% coverage)
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk and may not be suitable for all investors. Past performance is not indicative of future results.
