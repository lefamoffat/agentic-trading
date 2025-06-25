# Getting Started

This guide provides step-by-step instructions to set up the Agentic Trading project, install dependencies, and run the basic workflows.

## 📋 Prerequisites

-   Python 3.11+
-   [uv](https://github.com/astral-sh/uv) package manager
-   API keys for data providers (e.g., forex.com)

## 🛠️ Installation

1.  **Clone the repository**

    ```bash
    git clone <repository-url>
    cd agentic-trading
    ```

2.  **Install dependencies**

    Use `uv` to sync the project's dependencies from `pyproject.toml`.

    ```bash
    uv sync
    ```

3.  **Initialize the project**

    This script sets up the necessary directories for data, logs, and models.

    ```bash
    uv run python scripts/setup/init_project.py
    ```

4.  **Set up environment variables**

    Copy the example `.env` file and add your credentials.

    ```bash
    cp .env.example .env
    ```

    Then, edit the `.env` file with your API keys and desired settings.

## 🎮 Quick Start

Follow these steps to run a complete training and evaluation cycle.

### 1. Configure ML Tracking

The system uses a **generic ML tracking abstraction** that supports multiple backends. Choose the appropriate backend for your needs:

**Development Setup (Default):**

```bash
# No configuration needed - uses in-memory tracking
# Perfect for development and testing
uv run agentic train --symbol "EUR/USD" --timesteps 10000
```

**Production Setup (Aim Backend):**

```bash
# Initialize Aim repository (first time only)
uv run aim init

# Configure environment to use Aim backend
export ML_TRACKING_BACKEND=aim
export ML_STORAGE_PATH=./aim_logs

# Start Aim UI server for experiment visualization
uv run aim up

# Now all training uses rich Aim tracking
uv run agentic train --symbol "EUR/USD" --timesteps 10000
```

**Key Points:**

-   **Generic Design**: Application code never references specific backends
-   **Environment-Based**: Backend selection via environment variables only
-   **Real Data**: All backends use real market data from forex.com, Yahoo Finance, etc.
-   **UI Access**: Aim UI available at [http://localhost:43800](http://localhost:43800) when using Aim backend

### 2. Prepare Training Data

Prepare market data using the new CLI commands with intelligent caching:

```bash
# Download real market data (with automatic caching for future runs)
uv run agentic download-data --symbol EUR/USD --timeframe 1h --bars 365

# Or use the complete pipeline (download → qlib conversion → feature engineering)
uv run agentic prepare-data --symbol EUR/USD --timeframe 1h --bars 365
```

**Data Sources**: The system fetches real historical price data from:

-   **forex.com broker API** (primary source for forex data)
-   **Yahoo Finance** (future integration for stocks/crypto)
-   **Custom data providers** (extensible architecture)

### 3. Train an RL Agent

Now you can run the training pipeline. This will automatically prepare real market data and log the run using the generic ML tracking system.

```bash
# Start a new training run (includes automatic data preparation)
uv run python scripts/training/train_agent.py --symbol "EUR/USD" --timeframe 1h --timesteps 20000 --days 365

# Or use the streamlined CLI approach
uv run agentic train --symbol "EUR/USD" --timeframe 1h --timesteps 20000
```

### 4. Optimize Hyperparameters

To find the best hyperparameters for an agent, use the optimization script. This will run multiple training trials with different hyperparameter combinations and track them using the generic ML tracking system.

```bash
uv run python scripts/training/optimize_agent.py --symbol "EUR/USD" --timeframe 1h --timesteps 5000 --trials 20

# Or use CLI for shorter experiments
uv run agentic optimize --symbol "EUR/USD" --trials 10
```

### 5. Monitor with Dashboard

Launch the multi-page dashboard to monitor your experiments and models:

```bash
# Launch the dashboard (works with any ML tracking backend)
uv run python scripts/dashboard/launch_dashboard.py

# Or with custom settings
uv run python scripts/dashboard/launch_dashboard.py --host 0.0.0.0 --port 8080 --debug
```

Access the dashboard at [http://localhost:8050](http://localhost:8050).

The dashboard provides:

-   **Overview**: System status and performance metrics
-   **Experiments**: Browse and compare training runs with any ML tracking backend
-   **Models**: Manage trained models and trigger live trading
-   **Data Pipeline**: Monitor data quality and sources

### 6. One-Command Quickstart (CLI)

Prefer the Typer-powered `agentic` CLI for a zero-boilerplate workflow that bundles all steps (project init → data preparation → training → simulation) under a single command:

```bash
uv run agentic quickstart --symbol "EUR/USD" --timeframe 1h --timesteps 20000
```

The command will:

1. create the required directory structure (`agentic init`),
2. download and process real historical market data,
3. engineer Qlib features,
4. train the default PPO agent for the specified timesteps, and
5. launch the dashboard so you can inspect the trained model.

Advanced users can invoke individual sub-commands, e.g. `agentic train --help`, etc. Run `agentic --help` for the full list.

## 🧪 Testing the System

### Test ML Tracking System

```bash
# Test the ML tracking system (uses real market data)
uv run python scripts/demo_tracking.py
```

### Run Test Suite

Our comprehensive test suite follows strict patterns and includes robust health monitoring:

```bash
# Run by test type (following @testing.mdc markers)
uv run pytest -m unit                    # Pure logic tests, <50ms
uv run pytest -m component               # Component tests with mocks, <1s
uv run pytest -m integration             # Integration tests with real backends

# Run by location
uv run pytest src/*/tests                # Unit/component tests (fast)
uv run pytest integration_tests/         # Integration tests (comprehensive)

# Health monitoring validation (catches schema issues automatically)
uv run pytest integration_tests/test_backend_schema_validation.py
```

### Key Testing Features

-   ✅ **Automated Issue Detection**: Would catch SQLite schema issues like "no such table: run"
-   ✅ **Health Monitoring**: Comprehensive backend health validation
-   ✅ **Generic Testing**: Uses proper interfaces, no backend lock-in
-   ✅ **CLI Integration**: Complete workflow testing with error detection

## 🚀 Next Steps

Once you have the system running:

1. **Experiment with different agents**: Try PPO, A2C, or SAC algorithms
2. **Customize the environment**: Modify `configs/trading_config.yaml` for different reward systems
3. **Add new data sources**: Integrate additional market data providers
4. **Explore the dashboard**: Use the web interface to analyze your experiments
5. **Set up live trading**: Configure broker integration for real trading

## 💡 Tips for Success

### Development Workflow

1. **Use Aim for Tracking**: Aim provides rich experiment tracking and visualization
2. **Use CLI Commands**: The `agentic` CLI provides the most streamlined experience
3. **Monitor via Dashboard**: Keep the dashboard open to track experiment progress
4. **Experiment with Config**: Modify YAML files to try different trading strategies

### Production Setup

1. **Configure Aim**: Set up Aim for rich experiment tracking and visualization
2. **Set Up Monitoring**: Use the dashboard's system health monitoring
3. **Configure Risk Management**: Set appropriate limits in trading configuration
4. **Regular Backtesting**: Validate models on historical data before live trading

### Understanding the System

-   **Market Data**: Always real data from forex.com, Yahoo Finance, etc.
-   **ML Tracking**: Aim provides rich experiment analysis and visualization
-   **Messaging**: Real-time communication for live experiment monitoring
-   **Environment**: Dynamic trading simulation with configurable features
-   **Dashboard**: Unified interface combining live updates with historical analysis

**Key Documentation:**

-   [Experiment Data Management Guide](../docs/experiment_data_management.md) - How messaging and tracking work together
-   [Messaging System Guide](../src/messaging/README.md) - Real-time communication system
-   [Tracking System Guide](../src/tracking/README.md) - Persistent experiment logging

### Troubleshooting

-   **Data Issues**: Check `data/` directory permissions and disk space
-   **Training Errors**: Monitor logs in `logs/` directory
-   **Dashboard Problems**: Ensure port 8050 is available
-   **ML Tracking Issues**: Check backend configuration and storage paths
-   **Broker Connection**: Verify API credentials in `.env` file
