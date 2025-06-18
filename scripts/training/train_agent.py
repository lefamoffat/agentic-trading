#!/usr/bin/env python3
"""Train a reinforcement learning agent.

This script handles the training process for an RL agent, including
loading data, setting up the environment, training the agent, and
saving the trained model.

Usage:
    python -m scripts.training.train_agent [options]
"""
import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from src.callbacks.interrupt_callback import GracefulShutdownCallback
from src.callbacks.metrics_callback import MlflowMetricsCallback
from src.data.pipelines import run_data_preparation_pipeline
from src.environments.factory import environment_factory
from src.models.sb3.factory import agent_factory
from src.models.sb3.wrapper import Sb3ModelWrapper
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger
from src.utils.mlflow import (
    log_params,
    log_sb3_model,
    ensure_experiment,
)

logger = get_logger(__name__)

# Use stdlib distutils to prevent setuptools replacement warnings
os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")

# Ensure the experiment exists (single source of truth)
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "AgenticTrading")
ensure_experiment(EXPERIMENT_NAME)

def load_and_preprocess_data(symbol: str, timeframe: str) -> (pd.DataFrame, pd.DataFrame):
    """Load and preprocess the feature data, splitting it into train and eval sets."""
    sanitized_symbol = symbol.replace("/", "")
    features_path = Path(f"data/processed/features/{sanitized_symbol}_{timeframe}_features.csv")

    logger.info(f"Loading data from {features_path}...")
    try:
        data_df = pd.read_csv(features_path)
        # The timestamp column is loaded as a string, convert it to datetime
        data_df["timestamp"] = pd.to_datetime(data_df["timestamp"])
        data_df.set_index("timestamp", inplace=True)
        data_df.dropna(inplace=True)
        data_df.reset_index(drop=True, inplace=True)
    except FileNotFoundError:
        logger.error(f"Feature file not found at {features_path}. Please run the data preparation pipeline.")
        return None, None

    train_size = int(len(data_df) * 0.8)
    train_df, eval_df = data_df.iloc[:train_size], data_df.iloc[train_size:]
    logger.info(f"Training data: {len(train_df)} samples, Evaluation data: {len(eval_df)} samples")
    return train_df, eval_df


def train_agent(
    agent_name: str,
    train_env: Monitor,
    eval_env: Monitor,
    agent_config: Dict[str, Any],
    training_config: Dict[str, Any],
    timesteps: int,
) -> (BaseAlgorithm, BaseCallback):
    """Creates, trains, and returns an RL agent.

    Args:
        agent_name: The name of the agent to create.
        train_env: The environment for training.
        eval_env: The environment for evaluation.
        agent_config: Configuration parameters for the agent.
        training_config: Configuration parameters for training.
        timesteps: The total number of timesteps for training.

    Returns:
        A tuple containing the trained model and the callback list.

    """
    # derive current run ID automatically
    active_run = mlflow.active_run()
    run_id = active_run.info.run_id if active_run else "no_run"
    model_dir = Path("data/models") / agent_name / run_id
    log_dir = Path("logs/tensorboard") / agent_name / run_id

    # 1. Create Callbacks
    metrics_callback = MlflowMetricsCallback(
        eval_env=eval_env,
        eval_freq=training_config.get("eval_frequency", 500),
        n_eval_episodes=training_config.get("n_eval_episodes", 5),
        best_model_save_path=str(model_dir / "best_model.zip"),
    )
    shutdown_callback = GracefulShutdownCallback()
    callback_list = CallbackList([metrics_callback, shutdown_callback])

    # 2. Create Agent using the factory
    model = agent_factory.create_agent(
        name=agent_name,
        env=train_env,
        hyperparams=agent_config,
        tensorboard_log_path=str(log_dir),
    )

    # 3. Train Agent
    model.train(
        total_timesteps=timesteps,
        callback=callback_list,
    )

    return model, callback_list


def train_agent_session(
    agent_name: str,
    symbol: str,
    timeframe: str,
    timesteps: int,
    initial_balance: int,
    run_id: Optional[str] = None,
    agent_params: Optional[Dict[str, Any]] = None,
):
    """Orchestrates the agent training and evaluation process for a given session.

    Args:
        agent_name (str): Name of the agent to train (e.g., 'PPO').
        symbol (str): Trading symbol (e.g., 'EUR/USD').
        timeframe (str): Timeframe for the data (e.g., '1d').
        timesteps (int): Total number of timesteps for training.
        initial_balance (int): The initial balance for the trading environment.
        run_id: Optional existing MLflow run ID to resume.
        agent_params: Optional dictionary of agent parameters to override config.

    """
    # 1. Setup
    active_run = mlflow.active_run()
    if not active_run:
        raise RuntimeError(
            "train_agent_session must be called within an active MLflow run."
        )
    current_run_id = active_run.info.run_id
    logger.info(f"Executing within MLflow Run ID: {current_run_id}")

    config_loader = ConfigLoader()
    config = config_loader.load_config("agent_config")

    # Override agent parameters if provided
    if agent_params:
        agent_config = config.get(agent_name.lower(), {})
        agent_config.update(agent_params)
        config[agent_name.lower()] = agent_config
    else:
        agent_config = config.get(agent_name.lower(), {})

    training_config = config.get("training", {})
    logger.info(f"Starting training session for {agent_name} on {symbol} ({timeframe})...")

    # Log all agent parameters
    log_params(agent_config)
    mlflow.log_param("agent", agent_name)
    mlflow.log_param("symbol", symbol)
    mlflow.log_param("timeframe", timeframe)
    mlflow.log_param("timesteps", timesteps)

    # 2. Data Preparation
    logger.info("Starting full data preparation pipeline...")
    if not run_data_preparation_pipeline(symbol, timeframe):
        logger.error("Data preparation pipeline failed.")
        mlflow.end_run(status="FAILED")
        return

    # 3. Load Data
    train_df, eval_df = load_and_preprocess_data(symbol, timeframe)
    if train_df is None or eval_df is None:
        mlflow.end_run(status="FAILED")
        return

    # 4. Create Environments
    logger.info("Creating training and evaluation environments...")
    train_env = Monitor(environment_factory.create_environment(
        name="default",
        data=train_df,
        initial_balance=initial_balance,
        trade_fee=0.0,
    ))
    eval_env = Monitor(environment_factory.create_environment(
        name="default",
        data=eval_df,
        initial_balance=initial_balance,
        trade_fee=0.0,
    ))
    logger.info("Environments created successfully.")

    # 5. Agent Training
    logger.info(f"Training {agent_name} agent...")
    model, callback = train_agent(
        agent_name=agent_name,
        train_env=train_env,
        eval_env=eval_env,
        agent_config=agent_config,
        training_config=training_config,
        timesteps=timesteps,
    )
    logger.info("Agent training completed.")

    # 6. Model Saving
    final_model_path = Path("data/models") / agent_name / current_run_id / "final_model.zip"
    logger.info(f"Saving final model to {final_model_path}...")
    model.save(final_model_path)

    # Log the model as a *LoggedModel* (MLflow 3)
    logger.info("Logging and registering the model with MLflow...")

    example_df = train_df.head(1)

    log_sb3_model(
        model=model,
        name=f"{agent_name}_{symbol.replace('/', '')}",
        artifacts={"model_path": str(final_model_path)},
        signature_df=example_df,
        python_model=Sb3ModelWrapper(policy_name=agent_name),
    )

    logger.info(f"Training run {current_run_id} complete.")
    logger.info(f"Model saved at: {final_model_path}")
    logger.info(f"To view logs, run: tensorboard --logdir {Path('logs/tensorboard') / agent_name / current_run_id}")
    logger.info("To view MLflow UI, run: mlflow ui")


def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(description="Train an RL trading agent")
    parser.add_argument("--agent", type=str, default="PPO", help="Agent to train (e.g., PPO)")
    parser.add_argument("--symbol", type=str, default="EUR/USD", help="Symbol to train on")
    parser.add_argument("--timeframe", type=str, default="1d", help="Timeframe of the data")
    parser.add_argument("--timesteps", type=int, default=10000, help="Number of timesteps to train for")
    parser.add_argument("--balance", type=float, default=10000, help="Initial account balance")
    args = parser.parse_args()

    print("🚀 Starting Agent Training")
    print("========================================")
    print("Run ID: (new run will be created)")
    print(f"Agent: {args.agent}")
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Initial Balance: {args.balance}")
    print("========================================")

    run_name = f"{args.agent}_{args.symbol.replace('/', '')}_{args.timeframe}_{args.timesteps}"

    with mlflow.start_run(run_name=run_name) as run:
        train_agent_session(
            agent_name=args.agent,
            symbol=args.symbol,
            timeframe=args.timeframe,
            timesteps=args.timesteps,
            initial_balance=args.balance,
            run_id=run.info.run_id,
            agent_params=None,
        )


if __name__ == "__main__":
    main()
