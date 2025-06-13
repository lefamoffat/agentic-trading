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
from datetime import datetime
from pathlib import Path
from typing import Optional
import subprocess

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import EvalCallback

from src.agents.factory import agent_factory
from src.environments.factory import environment_factory
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger


def train_agent(
    agent_name: str,
    symbol: str,
    timeframe: str,
    total_timesteps: int,
    run_id: Optional[str] = None,
) -> None:
    """
    Train an RL agent with evaluation and TensorBoard logging.

    Args:
        agent_name (str): The name of the agent to train (e.g., "PPO").
        symbol (str): The trading symbol to use for training data (e.g., "EURUSD").
        timeframe (str): The timeframe of the training data (e.g., "1h").
        total_timesteps (int): The number of timesteps to train for.
        run_id (Optional[str]): The ID of a previous run to resume training from.
    """
    logger = get_logger(__name__)
    config_loader = ConfigLoader()
    agent_config = config_loader.load_config("agent_config")
    training_config = agent_config.get("training", {})

    # --- Step 0: Build Features ---
    logger.info("Building features with Qlib...")
    try:
        subprocess.run(
            [
                "python",
                "-m",
                "scripts.features.build_features",
                "--symbol",
                symbol,
                "--timeframe",
                timeframe,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Feature building failed: {e}")
        return
    except FileNotFoundError:
        logger.error(
            "Could not find the feature building script. Make sure you are in the project root."
        )
        return

    # --- Setup Run ID and Paths ---
    if run_id:
        logger.info(f"Resuming training for run ID: {run_id}")
    else:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        logger.info(f"Starting new training run with ID: {run_id}")
    
    model_dir = Path(f"data/models/{agent_name}/{run_id}")
    log_dir = Path(f"logs/tensorboard/{agent_name}/{run_id}")
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    best_model_path = model_dir / "best_model.zip"
    final_model_path = model_dir / "final_model.zip"

    # 1. Load and Split Data
    sanitized_symbol = symbol.replace("/", "")
    data_path = Path(f"data/processed/features/{sanitized_symbol}_{timeframe}_features.csv")
    if not data_path.exists():
        logger.error(f"Feature file not found at {data_path}. Please generate features first.")
        return
    
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, index_col="timestamp", parse_dates=True)
    
    # Time-based split for training and validation (80/20)
    train_size = int(len(df) * 0.8)
    train_df, eval_df = df[:train_size], df[train_size:]
    logger.info(f"Training data: {len(train_df)} samples")
    logger.info(f"Evaluation data: {len(eval_df)} samples")

    # 2. Create Environments
    logger.info("Creating training and evaluation environments...")
    train_env = environment_factory.create_environment("default", data=train_df)
    eval_env = environment_factory.create_environment("default", data=eval_df)

    # 3. Create Agent
    logger.info(f"Creating agent: {agent_name}")
    agent = agent_factory.create_agent(name=agent_name, env=train_env)
    
    # Load model if resuming a run
    if run_id and best_model_path.exists():
        logger.info(f"Loading best model from previous run: {best_model_path}")
        agent.load(best_model_path)

    # 4. Set up Callbacks
    logger.info("Setting up evaluation callback...")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=training_config.get("eval_frequency", 500),
        n_eval_episodes=training_config.get("eval_episodes", 5),
        deterministic=True,
        render=False,
        verbose=1,
    )
    
    # 5. Train Agent
    logger.info(f"Starting training for {total_timesteps} timesteps...")
    model_params = agent_config.get(agent_name.lower(), {})
    agent.train(
        total_timesteps=total_timesteps,
        model_params=model_params,
        callback=eval_callback,
        tensorboard_log_path=str(log_dir),
    )

    # 6. Save Final Model
    logger.info(f"Saving final model to {final_model_path}...")
    agent.save(final_model_path)
    logger.info(f"Training run {run_id} complete.")
    logger.info(f"Best model saved at: {best_model_path}")
    logger.info(f"To view logs, run: tensorboard --logdir {log_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train an RL trading agent")
    parser.add_argument(
        "--agent",
        type=str,
        default="PPO",
        help="The name of the agent to train (default: PPO)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="EUR/USD",
        help="The trading symbol to use (e.g., EUR/USD)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="The timeframe for the data (e.g., 1h)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=20000,
        help="The number of timesteps to train for (default: 20000)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="The ID of a previous run to resume training from (e.g., '20250613-122345')",
    )

    args = parser.parse_args()

    # Sanitize symbol for file paths
    sanitized_symbol = args.symbol.replace("/", "")

    print("🚀 Starting Agent Training")
    print("=" * 40)
    print(f"Agent: {args.agent}")
    print(f"Symbol: {args.symbol} (Sanitized: {sanitized_symbol})")
    print(f"Timeframe: {args.timeframe}")
    print(f"Timesteps: {args.timesteps}")
    if args.run_id:
        print(f"Resuming from Run ID: {args.run_id}")
    print("=" * 40)

    train_agent(
        agent_name=args.agent,
        symbol=sanitized_symbol,
        timeframe=args.timeframe,
        total_timesteps=args.timesteps,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main() 