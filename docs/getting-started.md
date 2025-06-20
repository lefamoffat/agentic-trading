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

### 2. Build Features with Qlib

Before training, you must generate features from the raw data using Qlib. This script downloads the latest data and calculates a predefined set of technical indicators.

```bash
uv run python scripts/features/build_features.py --symbol "EUR/USD" --timeframe 1h
```

### 3. Train an RL Agent

Now you can run the training pipeline. This will use the features generated in the previous step and log the run to MLflow if the server is active.

```bash
# Start a new training run
uv run python scripts/training/train_agent.py --symbol "EUR/USD" --timeframe 1h --timesteps 20000
```

### 4. Optimize Hyperparameters

To find the best hyperparameters for an agent, use the optimization script. This will run multiple training trials with different hyperparameter combinations and log them as nested runs in MLflow.

```bash
uv run python scripts/training/optimize_agent.py --symbol "EUR/USD" --timeframe 1h --timesteps 5000 --trials 20
```

### 5. One-Command Quickstart (CLI)

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
