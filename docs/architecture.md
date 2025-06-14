# Project Architecture Philosophy

This document outlines the core architectural principles of the Agentic Trading project to ensure consistent and maintainable development.

## Core Libraries and Their Roles

This project uses a hybrid architecture that leverages the strengths of two primary libraries: **Microsoft Qlib** and **Stable-Baselines3**. All new development should respect the designated role of each library.

### 1. Microsoft Qlib: The Quantitative Analysis Engine

**Qlib is the designated framework for all data-centric operations.** Its responsibilities include:

-   **Data Handling**: Managing financial datasets, including storage, format conversion, and calendar alignment.
-   **Feature Engineering**: Serving as the **sole engine** for calculating technical indicators and alpha factors.
-   **Backtesting**: Evaluating the performance of trained models and strategies at the portfolio level.

Do not re-implement data processing or feature calculation logic that can be handled by Qlib.

### 2. Stable-Baselines3: The Reinforcement Learning Engine

**Stable-Baselines3 is the designated framework for all reinforcement learning operations.** Its responsibilities include:

-   **RL Algorithms**: Providing robust, pre-built implementations of RL algorithms (PPO, SAC, etc.).
-   **Agent Training**: Managing the core training loop via the `learn()` method.
-   **Policy Representation**: Defining and storing the neural network policies that the agent learns.

### 3. Custom Code: The "Glue"

Custom code written in the `src/` directory should primarily serve as the **"glue"** that connects the two core libraries and adapts them to our specific use case. The primary example of this is:

-   **`src/environments`**: This module is a custom implementation of a `gymnasium`-compatible environment. It is necessary because Qlib does not provide a step-by-step, interactive environment suitable for RL agent training. This module acts as the bridge between Qlib's data output and Stable-Baselines3's agent input.
-   **`src/brokers`**: The broker-agnostic layer for connecting to live trading APIs.

## Development Golden Rule

Before writing new code, ask the following question:

> "Does this functionality rightfully belong in Qlib's world (data, features, backtesting), Stable-Baselines3's world (RL algorithms, training), or is it the necessary 'glue' that connects them?"

Adhering to this principle will ensure a clean, maintainable, and powerful architecture that leverages the best of both worlds.
