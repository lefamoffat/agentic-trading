# Agentic Trading - EUR/USD RL Trading System

A sophisticated multi-asset reinforcement learning trading system with broker-agnostic architecture, built with Microsoft Qlib, PyTorch, and modern ML practices.

## 🎯 Project Overview

This project implements a reinforcement learning-based trading system with modular, broker-agnostic architecture supporting multiple asset classes. It leverages:

-   **Microsoft Qlib** for quantitative trading infrastructure
-   **Broker-Agnostic Design** with factory patterns for easy broker addition
-   **Standardized Data Pipeline** ensuring consistent format across all sources
-   **Modular Market Calendars** supporting forex, stocks, crypto trading sessions
-   **Multiple RL Algorithms** (PPO, A3C, SAC) with automatic selection
-   **Multi-timeframe Analysis** (5m, 15m, 1h, 4h, daily)
-   **Live Trading Integration** with forex.com (Alpaca, others planned)
-   **Comprehensive Backtesting** and performance analysis

## 🚀 Features

-   ✅ **Broker-Agnostic Architecture** - Modular design supporting multiple brokers (forex.com, future: Alpaca, etc.)
-   ✅ **Standardized Data Pipeline** - Consistent CSV format across all data sources
-   ✅ **Market Calendar System** - Modular calendar support (forex 24/5, stocks, crypto)
-   ✅ **Data Quality Assurance** - Automatic validation and quality scoring
-   ✅ **Multi-Timeframe Support** - 5m, 15m, 1h, 4h, daily analysis
-   ✅ **Feature Engineering Framework** - Technical indicators and market analysis
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
FOREX_COM_USERNAME=your_username_here
FOREX_COM_PASSWORD=your_password_here
FOREX_COM_APP_KEY=your_app_key_here
FOREX_COM_SANDBOX=false
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
│   ├── brokers/           # Broker integrations (forex.com, future: Alpaca, etc.)
│   ├── data/              # Data handling & market calendars
│   │   ├── calendars/     # Market calendar system (forex, stocks, crypto)
│   │   └── processor.py   # Data standardization pipeline
│   ├── features/          # Feature engineering framework
│   ├── strategies/        # Trading strategy framework
│   ├── analysis/          # Market analysis tools
│   ├── backtesting/       # Backtesting framework
│   ├── trading/           # Live trading
│   ├── training/          # RL training
│   └── utils/             # Utilities
├── scripts/               # Executable scripts
│   ├── data/              # Data download & processing
│   ├── features/          # Feature generation
│   └── strategies/        # Strategy development
├── data/                  # Data storage (standardized CSV format)
├── logs/                  # Log files
├── results/               # Results and reports
└── tests/                 # Test suite
```

## 🎮 Quick Start

### 1. Download Historical Data

```bash
# Download EUR/USD data from forex.com
python -m scripts.data.download_historical --symbol "EUR/USD" --timeframe 1h --bars 365 --broker forex.com

# Download GBP/USD data
python -m scripts.data.download_historical --symbol "GBP/USD" --timeframe 4h --bars 100 --broker forex.com
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
-   Broker-agnostic data download system
-   Standardized CSV format: `timestamp,open,high,low,close,volume`

**🔄 Phase 3 In Progress:**

-   Feature engineering framework
-   Technical indicators library
-   Strategy framework foundation
-   Market analysis components

**📋 Upcoming Phases:**

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
