#!/usr/bin/env python3
"""
Train a reinforcement learning agent.

This script handles the training process for an RL agent, including
loading data, setting up the environment, training the agent, and
saving the trained model.

Usage:
    python -m scripts.training.train_agent [options]
"""
import argparse
import mlflow
import pandas as pd
from pathlib import Path
from typing import Dict, Any

import cloudpickle
import stable_baselines3
from stable_baselines3.common.base_class import BaseAlgorithm

from src.agents.factory import agent_factory
from src.data.pipelines import run_data_preparation_pipeline
from src.environments.factory import environment_factory
from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader
from src.callbacks.metrics_callback import MlflowMetricsCallback
from stable_baselines3.common.monitor import Monitor


logger = get_logger(__name__)


def setup_mlflow(run_name: str):
    """Configure and start the MLflow tracking run."""
    mlflow.set_experiment("agentic-trading")
    run = mlflow.start_run(run_name=run_name)
    logger.info(f"MLflow Run ID: {run.info.run_id}")
    return run.info.run_id


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


def train_agent_session(
    run_id: str,
    agent_name: str,
    symbol: str,
    timeframe: str,
    timesteps: int,
    initial_balance: float,
    hyperparam_overrides: Dict[str, Any] = None,
):
    """
    Orchestrates the agent training and evaluation process for a given session.

    Args:
        run_id (str): The ID of the MLflow run for tracking.
        agent_name (str): Name of the agent to train (e.g., 'PPO').
        symbol (str): Trading symbol (e.g., 'EUR/USD').
        timeframe (str): Timeframe for the data (e.g., '1d').
        timesteps (int): Total number of timesteps for training.
        initial_balance (float): The initial balance for the trading environment.
        hyperparam_overrides (Dict[str, Any], optional): Hyperparameters to override config.
    """
    log_dir = Path("logs/tensorboard") / agent_name / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Run Data Pipeline
    logger.info("Starting full data preparation pipeline...")
    if not run_data_preparation_pipeline(symbol, timeframe):
        logger.error("Data preparation pipeline failed.")
        mlflow.end_run(status="FAILED")
        return
        
    # 2. Load Data
    train_df, eval_df = load_and_preprocess_data(symbol, timeframe)
    if train_df is None or eval_df is None:
        mlflow.end_run(status="FAILED")
        return
        
    # 3. Create Environments
    logger.info("Creating training and evaluation environments...")
    train_env = Monitor(environment_factory.create_environment(
        name="default",
        data=train_df,
        initial_balance=initial_balance,
    ))
    eval_env = Monitor(environment_factory.create_environment(
        name="default",
        data=eval_df,
        initial_balance=initial_balance,
    ))

    # 4. Load Config and Set Hyperparameters
    config_loader = ConfigLoader()
    agent_config = config_loader.reload_config("agent_config")
    
    agent_params = agent_config.get(agent_name.lower(), {})
    if hyperparam_overrides:
        agent_params.update(hyperparam_overrides)
    
    training_params = agent_config.get("training", {})
    
    # Log hyperparameters to MLflow
    mlflow.log_params(agent_params)
    mlflow.log_params(training_params)

    # 5. Create Agent
    logger.info(f"Creating agent: {agent_name}")
    agent = agent_factory.create_agent(
        name=agent_name,
        env=train_env,
        hyperparams=agent_params,
        tensorboard_log_path=str(log_dir)
    )

    # 6. Set up Callbacks
    logger.info("Setting up evaluation callback...")
    model_dir = Path("data/models") / agent_name / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    
    eval_freq = training_params.get("eval_frequency", 500)
    n_eval_episodes = training_params.get("n_eval_episodes", 5)

    # Use the custom MLflow callback
    metrics_callback = MlflowMetricsCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes
    )

    # 7. Train Agent
    logger.info(f"Starting training for {timesteps} timesteps...")
    agent.train(
        total_timesteps=timesteps,
        callback=metrics_callback,
    )
    logger.info("Training complete.")

    # 8. Save and Log Final Model
    final_model_path = model_dir / "final_model.zip"
    logger.info(f"Saving final model to {final_model_path}...")
    agent.save(final_model_path)
    
    # Create a wrapper for the model for MLflow logging
    class Sb3ModelWrapper(mlflow.pyfunc.PythonModel):
        def __init__(self, agent_name):
            self.agent_name = agent_name
            self.model = None

        def load_context(self, context):
            model_class = getattr(stable_baselines3, self.agent_name)
            self.model = model_class.load(context.artifacts["model_path"])

        def predict(self, context, model_input):
            # SB3 expects a numpy array, not a pandas DataFrame
            obs = model_input.to_numpy()
            actions, _ = self.model.predict(obs, deterministic=True)
            return actions

    # Log the model to MLflow and register it
    logger.info("Logging and registering the model with MLflow...")
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=Sb3ModelWrapper(agent_name=agent_name),
        artifacts={"model_path": str(final_model_path)},
        registered_model_name=f"{agent_name}_{symbol.replace('/', '')}",
    )

    best_model_path = model_dir / "best_model.zip"
    logger.info(f"Training run {run_id} complete.")
    logger.info(f"Best model saved at: {best_model_path}")
    logger.info(f"To view logs, run: tensorboard --logdir {log_dir.parent.parent}")
    logger.info("To view MLflow UI, run: mlflow ui")
    mlflow.end_run()


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
    print(f"Agent: {args.agent}")
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Initial Balance: {args.balance}")
    print("========================================")
    
    run_name = f"{args.agent}_{args.symbol.replace('/', '')}_{args.timeframe}_{args.timesteps}"
    run_id = setup_mlflow(run_name)
    
    train_agent_session(
        run_id=run_id,
        agent_name=args.agent,
        symbol=args.symbol,
        timeframe=args.timeframe,
        timesteps=args.timesteps,
        initial_balance=args.balance,
        hyperparam_overrides=None, # No overrides for direct training
    )


if __name__ == "__main__":
    main()