# 🤖 Agentic Trading System

A sophisticated reinforcement learning-based trading system that combines market data ingestion, feature engineering, agent training, and live trading execution. Built with modern Python tools and designed for flexibility and production deployment.

## ✨ Features

### 🧠 **Intelligent Trading Agents**

-   ✅ **Multiple RL Algorithms**: PPO, A2C, SAC with Stable-Baselines3
-   ✅ **Custom Trading Environment**: Realistic market simulation with configurable parameters
-   ✅ **Advanced Reward Systems**: PnL-based, risk-adjusted, and composite reward functions
-   ✅ **Portfolio Management**: Multi-asset position sizing and risk management

### 📊 **Comprehensive Data Pipeline**

-   ✅ **Multi-Source Data**: Support for Forex.com, Yahoo Finance, and custom data sources
-   ✅ **Real-time Processing**: Live data ingestion with configurable timeframes
-   ✅ **Feature Engineering**: Technical indicators, market regime detection, and custom features
-   ✅ **Data Storage**: Efficient binary storage with Qlib integration

### 🔬 **Experiment Tracking & Analysis**

-   ✅ **ML Tracking**: Backend-agnostic experiment tracking (Aim, W&B, TensorBoard, Mock)
-   ✅ **Rich Metrics**: Performance analytics, risk metrics, and trading statistics
-   ✅ **Interactive Dashboard**: Real-time monitoring and experiment visualization
-   ✅ **Model Versioning**: Automatic model artifact management and deployment

### 🚀 **Production-Ready Infrastructure**

-   ✅ **Live Trading**: Real broker integration with Forex.com API
-   ✅ **Risk Management**: Position limits, stop-loss, and drawdown protection
-   ✅ **Monitoring**: System health checks, alerting, and performance tracking
-   ✅ **Scalability**: Containerized deployment with Docker and cloud-ready architecture

## 🚀 Quick Start

### Prerequisites

-   **Python 3.11+**
-   **uv** (Python package manager)
-   **Git** for version control

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-username/agentic-trading.git
cd agentic-trading

# Install dependencies
uv sync

# Activate environment (optional - uv run handles this automatically)
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

### 2. Quick Environment Setup

```bash
# Initialize project structure
uv run python scripts/setup/init_project.py

# Copy environment configuration
cp .env.example .env
# Edit .env with your API keys and settings
```

### 3. Initialize ML Tracking (Optional, for Rich Experiment Analysis)

The system uses **Aim** for ML experiment tracking. For production usage with detailed experiment visualization, you can optionally initialize and start the Aim UI server.

```bash
# Initialize Aim repository (first time only, for production tracking)
uv run aim init

# Start Aim UI server for experiment visualization (optional)
uv run aim up
```

You can access the Aim UI at [http://localhost:43800](http://localhost:43800) for detailed experiment analysis and comparison.

**Note**: The system uses **real market data** from forex.com, Yahoo Finance, etc. Aim handles experiment tracking and visualization.

### 4. Run Your First Training

Now you can run the training pipeline. This will automatically prepare data using the market_data module and log the run using the generic ML tracking system.

```bash
# Train a PPO agent on EUR/USD data
uv run python scripts/training/train_agent.py --agent ppo --symbol "EUR/USD" --timesteps 50000

# Or use the streamlined CLI approach
uv run agentic train --symbol "EUR/USD" --timeframe 1h --timesteps 20000
```

### 5. Hyperparameter Optimization

To find the best hyperparameters for an agent, use the optimization script. This will run multiple training trials with different hyperparameter combinations and track them using the generic ML tracking system.

```bash
# Optimize PPO hyperparameters
uv run python scripts/training/optimize_agent.py --agent ppo --symbol "EUR/USD" --n-trials 20

# Or use CLI for shorter experiments
uv run agentic optimize --symbol "EUR/USD" --trials 10
```

### 6. Launch Dashboard

Monitor your experiments and system performance through the interactive dashboard:

```bash
# Launch the dashboard (works with any ML tracking backend)
uv run python scripts/dashboard/launch_dashboard.py

# Optional: Start Aim UI for detailed experiment analysis (if using Aim backend)
uv run aim up

# Access dashboard at http://localhost:8050
# Aim UI available at http://localhost:43800 (if started)
```

### 7. One-Command Quickstart

For the fastest setup, use the comprehensive CLI command that handles everything:

```bash
# Complete workflow: setup → data → training → dashboard
uv run agentic quickstart --symbol "EUR/USD" --timeframe 1h --timesteps 20000
```

This command will:

1. Initialize the project structure
2. Download and process historical data from real sources (forex.com, etc.)
3. Train a PPO agent for the specified timesteps
4. Launch the dashboard for monitoring

## 📁 Project Structure

```
agentic-trading/
├── src/                          # Core source code
│   ├── agents/                   # RL agents and model wrappers
│   ├── environment/              # Dynamic trading environment simulation
│   │   └── sources/              # Real data sources (forex.com, Yahoo, etc.)
│   ├── tracking/                 # Generic ML tracking abstraction
│   │   ├── protocols.py          # Backend-agnostic interfaces
│   │   ├── models.py             # Trading-specific data models
│   │   ├── factory.py            # Backend switching system
│   │   └── backends/             # In-memory, Aim, and future backends
│   ├── brokers/                  # Live trading broker integrations
│   ├── training/                 # Training orchestration and services
│   ├── messaging/                # Event-driven messaging system
│   └── utils/                    # Shared utilities and helpers
├── apps/                         # Applications and interfaces
│   ├── dashboard/                # Multi-page web dashboard
│   └── cli/                      # Typer-powered CLI interface
├── scripts/                      # Automation and utility scripts
├── configs/                      # YAML configuration files
├── tests/                        # Unit and integration tests
└── docs/                         # Comprehensive documentation
```

## 🔧 Configuration

The system uses YAML configuration files for flexibility:

-   `configs/trading_config.yaml` - Trading environment and agent parameters
-   `configs/data_config.yaml` - Data sources and processing settings
-   `configs/broker_config.yaml` - Live trading broker configurations
-   `configs/messaging.yaml` - Event-driven messaging configuration

### Market Data Sources

All training uses **real market data** from configured sources:

```yaml
# configs/data_config.yaml
sources:
    forex_com:
        enabled: true
        sandbox: true # Use sandbox for development
    yahoo_finance:
        enabled: false # Future integration
```

## 🧪 Testing

Our comprehensive test suite follows strict testing patterns and would have automatically caught the backend schema issues we encountered:

```bash
# Run all tests
uv run pytest

# Run by test type (following @testing.mdc pattern)
uv run pytest -m unit                    # Pure logic tests, <50ms
uv run pytest -m component               # Component tests with mocks, <1s
uv run pytest -m integration             # Integration tests with real backends

# Run by location
uv run pytest src/*/tests                # Unit/component tests (fast)
uv run pytest integration_tests/         # Integration tests (comprehensive)

# Test ML tracking system
uv run python scripts/demo_tracking.py
```

### Test Coverage

Our test suite includes **comprehensive health monitoring and schema validation**:

-   ✅ **Backend Health Monitoring**: Detects database schema issues, corruption, dual database structures
-   ✅ **CLI Integration Testing**: Validates complete CLI workflow with health checks
-   ✅ **Schema Validation Tests**: Would catch "no such table: run" errors automatically
-   ✅ **Generic Interface Testing**: Uses proper testing patterns with no backend lock-in
-   ✅ **Error Recovery Testing**: Validates graceful degradation and recovery scenarios

## 📊 Monitoring & Analytics

### Dashboard Features

-   **Overview Page**: System status, recent experiments, performance trends, top models
-   **Experiments Page**: Browse and filter all training runs with comparison charts
-   **Single Experiment**: Detailed analysis with metrics, parameters, and training progress
-   **Single Model**: Model management with live trading controls and backtesting
-   **Data Pipeline**: Real-time monitoring of data sources and quality metrics

### ML Tracking Integration

-   **Automatic Logging**: All experiments tracked with metadata and metrics using Aim
-   **Rich Metrics**: Trading-specific metrics (Sharpe ratio, portfolio value, win rate)
-   **Artifact Management**: Model checkpoints and training artifacts
-   **Hyperparameter Tracking**: Complete parameter history and optimization
-   **Health Monitoring**: System health checks and performance diagnostics

### Real-time Features

-   Auto-refresh intervals (15-30 seconds) for live data updates
-   Interactive Plotly charts with filtering and sorting
-   Bootstrap-based responsive design
-   Graceful error handling and backend fallbacks

## 🔄 Development Workflow

1. **Data Preparation**: Configure data sources and run ingestion (`uv run agentic prepare-data`)
2. **Environment Setup**: Define trading rules and reward functions in YAML configs
3. **Agent Training**: Train RL agents with hyperparameter optimization (`uv run agentic train`)
4. **Experiment Analysis**: Monitor progress via dashboard or Aim UI
5. **Backtesting**: Validate performance on historical data
6. **Paper Trading**: Test with live data but simulated execution
7. **Live Deployment**: Deploy to production with risk management

## 🛠️ Advanced Usage

### Custom Agents

```python
from src.agents import BaseAgent
from stable_baselines3 import PPO

class CustomAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = PPO("MlpPolicy", self.env, **kwargs)
```

### Custom Environments

```python
from src.environment import TradingEnv, TradingEnvironmentConfig

config = TradingEnvironmentConfig(
    initial_balance=100000,
    observation_features=['close', 'volume', 'rsi', 'macd'],
    include_time_features=True,
    reward_system="sharpe_ratio"
)

env = TradingEnv(data=market_data, config=config)
```

### ML Tracking with Aim

```python
from src.tracking import get_ml_tracker, TrainingMetrics

# Get Aim tracker instance
tracker = await get_ml_tracker()
run = await tracker.start_run(experiment_id, config)

# Log trading-specific metrics
metrics = TrainingMetrics(
    reward=1250.0,
    portfolio_value=105000.0,
    sharpe_ratio=1.85,
    win_rate=0.67
)
await tracker.log_training_metrics(run.id, metrics, step)
```

### Live Trading

```python
from src.brokers import ForexComBroker

broker = ForexComBroker(
    api_key="your_api_key",
    account_id="your_account"
)
```

## 📈 Performance

The system is designed for high performance:

-   **Vectorized Operations**: NumPy and Pandas for efficient computation
-   **Async Processing**: Non-blocking I/O for data ingestion and API calls
-   **Intelligent Caching**: Cached data preparation and model predictions
-   **Parallel Training**: Multi-process training for hyperparameter optimization
-   **Event-Driven Architecture**: Real-time messaging with <100ms latency

## 🚀 Key Benefits

### Clean ML Tracking with Aim

-   **Rich Experiment Tracking**: Comprehensive experiment logging and visualization with Aim
-   **Trading-Specific Metrics**: Specialized metrics for financial applications
-   **Easy Development**: Clean interfaces for logging and querying experiments
-   **Production Ready**: Aim backend with rich visualizations and analysis
-   **Type Safety**: Full protocol-based typing with comprehensive error handling

### Dynamic Environment System

-   **Model Agnostic**: Works with any ML framework (SB3, PyTorch, LLMs, human input)
-   **Configurable Observations**: Any combination of market features and technical indicators
-   **Time-Aware Trading**: Respects market hours and trading sessions
-   **Flexible Actions**: Support for discrete, continuous, and string-based actions

### Real Market Data Integration

-   **Live Broker APIs**: Direct integration with forex.com and other brokers
-   **Multiple Sources**: Support for various data providers
-   **Quality Assurance**: Data validation and completeness checks
-   **Efficient Storage**: Qlib binary format for fast feature engineering

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure all tests pass (`

## Dashboard (Astro)

Real-time web UI powered by Astro/Preact/Tailwind. See `apps/dashboard/README.md` for setup and usage.
