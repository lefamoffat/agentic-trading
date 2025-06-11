# Agentic Trading - EUR/USD RL Trading System

A sophisticated EUR/USD reinforcement learning trading system built with Microsoft Qlib, PyTorch, and modern ML practices.

## 🎯 Project Overview

This project implements a reinforcement learning-based trading system specifically designed for EUR/USD forex trading. It leverages:

-   **Microsoft Qlib** for quantitative trading infrastructure
-   **Multiple RL Algorithms** (PPO, A3C, SAC) with automatic selection
-   **Multi-timeframe Analysis** (5m, 15m, 1h, 4h, daily)
-   **Live Trading Integration** with forex.com and Alpaca
-   **Comprehensive Backtesting** and performance analysis

## 🚀 Features

-   ✅ **Multi-Algorithm RL Training** - Automatic algorithm selection and hyperparameter optimization
-   ✅ **EUR/USD Focused** - Optimized specifically for major forex pair trading
-   ✅ **Multi-Source Data** - Integrates forex.com, Alpaca, Alpha Vantage, and Yahoo Finance
-   ✅ **Risk Management** - Built-in stop loss, take profit, and drawdown controls
-   ✅ **Live Trading Ready** - Production-ready broker integrations
-   ✅ **Configurable** - YAML-based configuration for all parameters
-   ✅ **Comprehensive Logging** - Detailed logging and monitoring

## 📋 Prerequisites

-   Python 3.11+
-   [uv](https://github.com/astral-sh/uv) package manager
-   API keys for data providers (forex.com, Alpaca, Alpha Vantage)

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
    python scripts/setup/init_project.py
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
FOREX_COM_API_KEY=your_key_here
FOREX_COM_API_SECRET=your_secret_here
ALPACA_API_KEY=your_key_here
ALPACA_API_SECRET=your_secret_here
ALPHA_VANTAGE_API_KEY=your_key_here

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
│   ├── core/              # Core trading logic
│   ├── data/              # Data handling
│   ├── backtesting/       # Backtesting framework
│   ├── trading/           # Live trading
│   ├── training/          # RL training
│   └── utils/             # Utilities
├── scripts/               # Executable scripts
├── data/                  # Data storage
├── logs/                  # Log files
├── results/               # Results and reports
└── tests/                 # Test suite
```

## 🎮 Quick Start

### 1. Download Historical Data

```bash
python scripts/data/download_historical.py
```

### 2. Train an RL Agent

```bash
python scripts/training/train_agent.py --algorithm ppo --timeframe 1h
```

### 3. Run Backtesting

```bash
python scripts/backtesting/run_backtest.py --model best_ppo_model
```

### 4. Start Paper Trading

```bash
python scripts/trading/start_live_trading.py --paper-trading
```

## 🧠 RL Algorithms

The system supports multiple RL algorithms:

-   **PPO (Proximal Policy Optimization)** - Stable and sample-efficient
-   **A3C (Asynchronous Actor-Critic)** - Good for parallel training
-   **SAC (Soft Actor-Critic)** - Excellent for continuous action spaces

### State Space

-   OHLCV data
-   RSI (14-period)
-   EMA (24 and 120 periods)
-   Normalized features

### Action Space

-   **-1**: Sell
-   **0**: Hold
-   **1**: Buy

### Reward Function

-   Profit/Loss based
-   Sharpe ratio optimization
-   Drawdown penalties

## 📈 Trading Features

### Risk Management

-   Maximum drawdown limits
-   Stop loss and take profit
-   Position sizing controls
-   Daily loss limits

### Timeframes

-   5 minutes
-   15 minutes
-   1 hour
-   4 hours
-   Daily

### Market Hours

-   Optimized for EUR/USD active hours (07:00-17:00 UTC)
-   Excludes weekends and holidays

## 🔍 Monitoring & Analysis

### Logging

-   Structured logging with loguru
-   Separate logs for training, backtesting, and trading
-   Configurable log levels

### Performance Metrics

-   Sharpe ratio
-   Maximum drawdown
-   Win rate
-   Profit factor
-   Sortino ratio

## 🧪 Development

### Running Tests

```bash
uv run pytest tests/
```

### Code Formatting

```bash
uv run black src/
```

### Type Checking

```bash
uv run mypy src/
```

## 📝 Phase 1 Status

**✅ Completed:**

-   Project structure and configuration
-   Core utilities (config loader, logging, settings)
-   YAML configuration files
-   Environment setup
-   Project initialization

**🔄 Next Phases:**

-   Data providers and processing
-   RL environment and agents
-   Backtesting framework
-   Live trading integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading forex involves substantial risk and may not be suitable for all investors. Past performance is not indicative of future results.
