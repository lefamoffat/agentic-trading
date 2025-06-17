#!/usr/bin/env python3
"""Hyperparameter optimization for an RL agent using Optuna and MLflow.

This script runs multiple training trials to find the best hyperparameters
for a given agent and environment.

Usage:
    python -m scripts.training.optimize_agent [options]
"""
import argparse
from typing import Any, Dict

import mlflow
import optuna

from scripts.training.train_agent import train_agent_session
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger
from src.utils.mlflow import log_params

logger = get_logger(__name__)

def suggest_hyperparameters(trial: optuna.Trial, hpo_config: Dict[str, Any]) -> Dict[str, Any]:
    """Suggest hyperparameters for a given trial based on the HPO config."""
    params = {}
    for param_name, config in hpo_config.items():
        if config["type"] == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, config["choices"])
        elif config["type"] == "int":
            params[param_name] = trial.suggest_int(param_name, config["low"], config["high"])
        elif config["type"] == "float":
            params[param_name] = trial.suggest_float(param_name, config["low"], config["high"])
        elif config["type"] == "log_float":
            params[param_name] = trial.suggest_float(param_name, config["low"], config["high"], log=True)
    return params

def objective(
    trial: optuna.Trial,
    agent_name: str,
    symbol: str,
    timeframe: str,
    timesteps: int,
    initial_balance: float,
    parent_run_id: str,
) -> float:
    """The objective function for Optuna to optimize."""
    # 1. Start a nested MLflow run for this trial
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True) as run:
        logger.info(f"--- Starting Trial {trial.number} (Run ID: {run.info.run_id}) ---")

        # 2. Suggest hyperparameters
        config_loader = ConfigLoader()
        hpo_config = config_loader.load_config("agent_config")["hpo_params"][agent_name.lower()]
        hyperparams = suggest_hyperparameters(trial, hpo_config)
        log_params(hyperparams)

        # 3. Run the training session with the suggested hyperparameters
        try:
            train_agent_session(
                agent_name=agent_name,
                symbol=symbol,
                timeframe=timeframe,
                timesteps=timesteps,
                initial_balance=initial_balance,
                agent_params=hyperparams,
            )
        except Exception as e:
            logger.error(f"Trial {trial.number} failed with error: {e}", exc_info=True)
            # Report failure to Optuna so it doesn't try this again
            raise optuna.exceptions.TrialPruned() from e

        # 4. Fetch the primary metric to optimize (e.g., Sharpe ratio)
        # We assume the last recorded value is from the final evaluation.
        metric_history = mlflow.tracking.MlflowClient().get_metric_history(run.info.run_id, "eval/sharpe_ratio")
        if not metric_history:
            logger.warning("Could not find 'eval/sharpe_ratio' metric for trial. Returning -1.0")
            return -1.0

        final_sharpe = metric_history[-1].value
        logger.info(f"--- Trial {trial.number} Finished. Sharpe Ratio: {final_sharpe:.4f} ---")
        return final_sharpe

def main():
    """Main entry point for the optimization script."""
    parser = argparse.ArgumentParser(description="Optimize an RL trading agent's hyperparameters.")
    parser.add_argument("--agent", type=str, default="PPO", help="Agent to optimize")
    parser.add_argument("--symbol", type=str, default="EUR/USD", help="Symbol to train on")
    parser.add_argument("--timeframe", type=str, default="1d", help="Timeframe of the data")
    parser.add_argument("--timesteps", type=int, default=5000, help="Number of timesteps per trial")
    parser.add_argument("--trials", type=int, default=20, help="Number of optimization trials to run")
    parser.add_argument("--balance", type=float, default=10000, help="Initial account balance")
    args = parser.parse_args()

    print("🚀 Starting Hyperparameter Optimization")
    print("========================================")
    print(f"Agent: {args.agent}")
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Timesteps per trial: {args.timesteps}")
    print(f"Number of trials: {args.trials}")
    print("========================================")

    # 1. Create a parent MLflow run for the entire optimization study
    run_name = f"HPO_{args.agent}_{args.symbol.replace('/', '')}_{args.timeframe}"
    with mlflow.start_run(run_name=run_name) as parent_run:
        logger.info(f"Parent MLflow Run ID for HPO Study: {parent_run.info.run_id}")

        # 2. Create the Optuna study
        study = optuna.create_study(direction="maximize")

        # 3. Run the optimization
        study.optimize(
            lambda trial: objective(
                trial,
                agent_name=args.agent,
                symbol=args.symbol,
                timeframe=args.timeframe,
                timesteps=args.timesteps,
                initial_balance=args.balance,
                parent_run_id=parent_run.info.run_id
            ),
            n_trials=args.trials,
            n_jobs=1 # Run trials sequentially
        )

        # 4. Log the best trial's results to the parent run
        mlflow.set_tag("best_trial_number", study.best_trial.number)
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metric("best_sharpe_ratio", study.best_value)

        logger.info("\n🎉 Optimization Finished!")
        logger.info(f"Best Trial: {study.best_trial.number}")
        logger.info(f"  Sharpe Ratio: {study.best_value:.4f}")
        logger.info("  Hyperparameters:")
        for key, value in study.best_trial.params.items():
            logger.info(f"    {key}: {value}")

if __name__ == "__main__":
    main()
