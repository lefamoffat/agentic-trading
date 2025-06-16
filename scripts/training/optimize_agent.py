#!/usr/bin/env python3
"""
Hyperparameter optimization for an RL agent using Optuna and MLflow.

This script uses Optuna to find the best hyperparameters for a given agent.
Each trial is logged as a nested run in MLflow.
"""
import argparse
import optuna
import mlflow
from pathlib import Path

from scripts.training.train_agent import train_agent_session
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)

def objective(
    trial: optuna.Trial,
    agent_name: str,
    symbol: str,
    timeframe: str,
    timesteps: int,
    initial_balance: float,
) -> float:
    """
    The objective function for Optuna to optimize.
    
    Args:
        trial: An Optuna Trial object.
        agent_name: The name of the agent to train.
        ... and other training params ...
        
    Returns:
        The metric to optimize (e.g., Sharpe ratio).
    """
    # Define the hyperparameter search space
    # This is a basic example for PPO. This should be expanded.
    hyperparams = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [256, 512, 1024, 2048]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }
    
    run_name = f"trial_{trial.number}_{agent_name}"
    
    with mlflow.start_run(nested=True, run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"Starting Trial {trial.number} with Run ID: {run_id}")
        
        # Log suggested hyperparameters
        mlflow.log_params(trial.params)
        
        # Here we would need to override the config for the agent
        # For now, we are not implementing the full override logic
        # but this is where it would go.
        
        train_agent_session(
            run_id=run_id,
            agent_name=agent_name,
            symbol=symbol,
            timeframe=timeframe,
            timesteps=timesteps,
            initial_balance=initial_balance,
            hyperparam_overrides=hyperparams,
        )
        
        # Fetch the metric to optimize from the MLflow run
        client = mlflow.tracking.MlflowClient()
        run_data = client.get_run(run_id).data
        metric_to_optimize = run_data.metrics.get("eval_sharpe_ratio", 0.0)
        
    return metric_to_optimize

def main():
    parser = argparse.ArgumentParser(description="Optimize an RL trading agent")
    parser.add_argument("--agent", type=str, default="PPO", help="Agent to optimize")
    parser.add_argument("--symbol", type=str, default="EUR/USD", help="Symbol")
    parser.add_argument("--timeframe", type=str, default="1d", help="Timeframe")
    parser.add_argument("--timesteps", type=int, default=10000, help="Timesteps per trial")
    parser.add_argument("--balance", type=float, default=10000, help="Initial balance")
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials")
    args = parser.parse_args()

    # Start a parent MLflow run for the optimization study
    mlflow.set_experiment("agentic-trading-hpo")
    with mlflow.start_run(run_name=f"{args.agent}_HPO"):
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(
                trial,
                args.agent,
                args.symbol,
                args.timeframe,
                args.timesteps,
                args.balance,
            ),
            n_trials=args.trials,
        )
        
        logger.info("Optimization finished.")
        logger.info(f"Best trial: {study.best_trial.value}")
        logger.info(f"Best params: {study.best_trial.params}")
        
if __name__ == "__main__":
    main() 