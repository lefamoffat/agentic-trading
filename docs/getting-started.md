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

### 1. Launch MLflow Server (Optional)

To track experiments, parameters, and metrics, first launch the MLflow tracking server. A helper script is provided.

```bash
bash scripts/setup/launch_mlflow.sh
```

You can access the MLflow UI at [http://localhost:5001](http://localhost:5001).

### 2. Prepare Training Data

Prepare market data using the centralized market_data module. This replaces the old multi-step process with a single, streamlined command:

```bash
# Prepare data using the new market_data module
uv run python scripts/data/prepare_data.py --symbol "EUR/USD" --timeframe 1h --days 365
```

### 3. Train an RL Agent

Now you can run the training pipeline. This will automatically prepare data using the market_data module and log the run to MLflow if the server is active.

```bash
# Start a new training run (includes automatic data preparation)
uv run python scripts/training/train_agent.py --symbol "EUR/USD" --timeframe 1h --timesteps 20000 --days 365
```

### 4. Optimize Hyperparameters

To find the best hyperparameters for an agent, use the optimization script. This will run multiple training trials with different hyperparameter combinations and log them as nested runs in MLflow.

```bash
uv run python scripts/training/optimize_agent.py --symbol "EUR/USD" --timeframe 1h --timesteps 5000 --trials 20
```

**Note:** The optimization script will be updated in a future version to use the new market_data module.

### 5. Monitor with Dashboard

Launch the multi-page dashboard to monitor your experiments and models:

```bash
# Launch the dashboard
uv run python scripts/dashboard/launch_dashboard.py

# Or with custom settings
uv run python scripts/dashboard/launch_dashboard.py --host 0.0.0.0 --port 8080 --debug
```

Access the dashboard at [http://localhost:8050](http://localhost:8050).

The dashboard provides:

-   **Overview**: System status and performance metrics
-   **Experiments**: Browse and compare training runs
-   **Models**: Manage trained models and trigger live trading
-   **Data Pipeline**: Monitor data quality and sources

### 6. One-Command Quickstart (CLI)

Prefer the Typer-powered `agentic` CLI for a zero-boilerplate workflow that bundles all steps (project init → data preparation → training → simulation) under a single command:

```bash
uv run agentic quickstart --symbol "EUR/USD" --timeframe 1h --timesteps 20000
```

The command will:

1. create the required directory structure (`agentic init`),
2. download and process historical data,
3. engineer Qlib features,
4. train the default PPO agent for the specified timesteps, and
5. launch the Streamlit dashboard so you can inspect the trained model.

Advanced users can invoke individual sub-commands, e.g. `agentic train --help`, etc. Run `agentic --help` for the full list.
