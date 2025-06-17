# Project Implementation Plan: The Ultimate Trading System

This document outlines the phased implementation plan to evolve the agentic trading project into a fully autonomous, cloud-native, and self-improving system.

## Guiding Principles

-   **Phased Approach**: We will implement features in logical phases, ensuring a stable and functional system at each stage.
-   **Automation First**: All infrastructure and deployment will be managed as code (Terraform) to eliminate manual configuration and ensure reproducibility.
-   **Leverage Existing Strengths**: We will continue to respect the architecture outlined in `docs/architecture.md`, using Qlib for data/backtesting and Stable-Baselines3 for RL, while our custom code acts as the "glue".

---

## Phase 0: Codebase Refactoring & Quality Enhancement (Completed)

**Goal:** Solidify the existing codebase to ensure it is robust, maintainable, and ready for future expansion. This phase addresses technical debt and enforces best practices.

**Key Achievements:**

-   **Broker Module Refactoring:**

    -   Overhauled the `forex_com` broker module to improve code quality and remove "hacky" logic.
    -   Introduced module-specific type definitions (`src/brokers/forex_com/types.py`) to eliminate hardcoded "magic strings" for API keys and parameters.
    -   Replaced ambiguous logic (e.g., `data.get("PriceBars") or data.get("BarHistory")`) with explicit, assertive code that relies on a single source of truth for API contracts.

-   **Improved Test Suite Consistency:**

    -   Aligned unit tests with integration tests by correcting mock data to reflect real-world API responses.
    -   Replaced hardcoded values in tests with enums (e.g., `Timeframe.H1.value` instead of `"1h"`), making tests more readable and less brittle.

-   **Enforced Type Safety:**
    -   Leveraged dataclasses and enums to establish a single source of truth for API-related constants, reducing the risk of typos and improving developer experience through autocompletion and static analysis.

This foundational work ensures that as we add more complex features in subsequent phases, we are building on a stable and high-quality platform.

---

## Phase 1: MLOps Foundation & Hyperparameter Optimization (HPO) (Completed)

**Goal:** Establish a robust system for tracking experiments and automatically finding optimal model hyperparameters. This is the foundation for the "smart" training sessions.

**Key Technologies:**

-   **MLflow**: For experiment tracking, model registry, and performance dashboards.
-   **Optuna**: For state-of-the-art hyperparameter optimization.

### Key Achievements:

-   **MLflow Integration:**
    -   Successfully integrated the `mlflow` library into the training pipeline.
    -   The training script (`scripts/training/train_agent.py`) now logs all critical information to a tracking server:
        -   Hyperparameters from `agent_config.yaml`.
        -   Custom performance metrics (Sharpe ratio, profit, drawdown) via a custom callback.
        -   The final trained model artifact, correctly packaged and registered in the MLflow Model Registry.
-   **Local MLflow Environment:**
    -   Created a `scripts/setup/launch_mlflow.sh` script to spin up a local, persistent MLflow server using Docker, streamlining the development workflow.
-   **Robust Data & Configuration Pipeline:**
    -   Systematically debugged and hardened the entire pipeline, from Qlib frequency handling to resolving ambiguous configuration files, ensuring reliable and repeatable runs.

### Next Steps: HPO Integration (Completed)

1.  **Integrate Optuna for HPO:**

    -   Added `optuna` and `optuna-integration` to the project dependencies.
    -   Created a new, robust `scripts/training/optimize_agent.py` script.
    -   The script defines an `objective` function that Optuna maximizes (Sharpe ratio).
    -   Each trial is logged as a nested run under a single parent MLflow experiment, allowing for easy comparison and visualization of results.
    -   The hyperparameter search space is now defined declaratively in `config/agent_config.yaml`, making it easy to configure.
    -   The optimization script reuses the core `train_agent_session` function, ensuring that HPO trials benefit from the same robust training logic as single runs.

---

## Phase 2: Interactive Simulation Mode (Completed)

**Goal:** Create a user-friendly web interface to visually backtest trained models against historical data, allowing for qualitative analysis and stress-testing.

**Key Technologies:**

-   **Streamlit**: For creating the web application UI.
-   **Plotly**: For interactive charting of trades and performance.
-   **Qlib**: As the backend for running the backtest.

### Implementation Steps:

1.  **Create Streamlit Application:**

    -   Create a new file `scripts/analysis/run_simulation.py`.
    -   The UI will connect to the MLflow server to fetch a list of registered models.
    -   Add a dropdown menu for the user to select a model and version.
    -   Add date pickers for selecting a historical simulation period.

2.  **Develop Simulation Backend:**

    -   The application will load the selected model from MLflow.
    -   It will use Qlib's backtesting engine to execute the model's strategy over the chosen historical data.
    -   The simulation will generate a trade log and performance statistics.

3.  **Visualize Results:**
    -   Use Plotly to create an interactive chart showing:
        -   The asset's price history.
        -   Buy/Sell markers at the points where the agent traded.
        -   An equity curve showing the agent's performance over time.
    -   Display key metrics (Total Profit, Sharpe Ratio, Max Drawdown) in the UI.

### Key Achievements:

-   **Streamlit UI** (`scripts/analysis/run_simulation.py`) delivers a browser-based workflow to pick any registered MLflow model, select a date range, and run a simulation with one click.
-   **Reusable Simulation Backend** (`src/simulation/`) provides a test-covered `Backtester` class and Plotly visualisation helpers that can be re-used by notebooks or scheduled reports.
-   **Full Test Coverage & Lint** – New unit tests (`src/simulation/tests/`) keep fast suite at 100 % pass rate; Ruff & mypy clean.
-   **Continuous Integration Ready** – Streamlit and Plotly added to dependency list; no regressions in CI gates.

---

## Phase 3: Cloud Deployment & Automation (AWS)

**Goal:** Deploy the entire system to AWS, enabling 24/7 live trading and automated training without relying on a local machine.

**Key Technologies:**

-   **Terraform**: For Infrastructure as Code (IaC).
-   **Docker**: For containerizing the application components.
-   **AWS**:
    -   **S3**: For the MLflow artifact store and historical data lake.
    -   **RDS (PostgreSQL)**: As the backend database for the MLflow server.
    -   **Fargate**: For serverless container execution of the MLflow server and the live trading agent.
    -   **Secrets Manager**: To securely store API keys and credentials.
    -   **IAM**: For managing permissions securely.
-   **GitHub Actions**: For CI/CD automation.

### Implementation Steps:

1.  **Create Terraform Configuration:**

    -   Set up a `terraform/` directory.
    -   Define all AWS resources as code: VPC, S3 buckets, RDS instance, Fargate services, IAM roles, etc.
    -   Ensure outputs are configured for easy access to resource names and endpoints.

2.  **Containerize the Application:**

    -   Create a `Dockerfile` that packages the application with all its dependencies.
    -   The container will be configurable via environment variables to run in different modes (e.g., `live-trader`, `training-job`).

3.  **Develop CI/CD Pipeline (GitHub Actions):**
    -   Create a `.github/workflows/deploy.yml` file.
    -   The pipeline will trigger on pushes to the `main` branch.
    -   **Steps:**
        1.  Run tests (`pytest`).
        2.  Build and push the Docker image to AWS ECR (Elastic Container Registry).
        3.  Run `terraform apply` to deploy any infrastructure changes.
        4.  Update the AWS Fargate service to use the new Docker image.

---

## Phase 4: Future Goal - Advanced "Smart" Training

**Goal:** Evolve the HPO system into a meta-learning controller that can reason about past experiments and manage different trading styles autonomously.

### Exploration Paths:

1.  **LLM-Driven HPO:**

    -   Develop a controller that periodically fetches all experiment data from the MLflow server.
    -   Feed this data into an LLM with a prompt designed to analyze the results and suggest a new, more promising search space for Optuna.
    -   Example Prompt: _"Given these results, which hyperparameter ranges should we explore next to improve the Sharpe ratio for a day-trading model?"_

2.  **Multi-Model Management:**
    -   Extend the system to manage separate, independent training pipelines for different "trading styles" (e.g., 1-minute scalping, 4-hour swing trading).
    -   Each style would have its own set of Optuna studies and MLflow experiments, allowing the system to optimize them independently.
