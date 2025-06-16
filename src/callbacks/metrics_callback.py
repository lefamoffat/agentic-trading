#!/usr/bin/env python3
"""
Custom callbacks for Stable-Baselines3.
"""
from typing import Dict, Any

import numpy as np
import mlflow
from stable_baselines3.common.callbacks import BaseCallback

from src.callbacks.utils import calculate_performance_metrics
from src.environments.wrappers import EvaluationWrapper

class MlflowMetricsCallback(BaseCallback):
    """
    A custom callback to log evaluation metrics to MLflow and save the best model.
    
    This callback runs policy evaluation at regular intervals and logs key
    performance metrics (Sharpe ratio, profit, drawdown) to the active
    MLflow run.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int,
        n_eval_episodes: int = 5,
        timeframe: str = "1d",
        best_model_save_path: str = None,
        verbose: int = 0
    ):
        super().__init__(verbose)
        # It's recommended to wrap the eval env here for clarity
        self.eval_env = EvaluationWrapper(eval_env)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.timeframe = timeframe
        self.best_model_save_path = best_model_save_path
        self.best_mean_sharpe = -np.inf

    def _on_step(self) -> bool:
        """
        This method is called after each step in the training process.
        """
        if self.n_calls % self.eval_freq == 0:
            self.run_evaluation()
        return True

    def run_evaluation(self):
        """
        Run policy evaluation and log the results to MLflow.
        """
        all_metrics = []
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated

            # After the episode, calculate metrics from the wrapper's data
            portfolio_values = self.eval_env.portfolio_values
            trade_history = self.eval_env.trade_history
            metrics = calculate_performance_metrics(
                portfolio_values,
                trade_history,
                self.timeframe
            )
            all_metrics.append(metrics)

        # Average the metrics over all evaluation episodes
        if all_metrics:
            avg_metrics = {
                key: np.mean([m[key] for m in all_metrics])
                for key in all_metrics[0]
            }
            
            # Log all metrics to the console and MLflow
            for key, value in avg_metrics.items():
                self.logger.record(f"eval/{key}", value)
            
            mlflow.log_metrics(avg_metrics, step=self.n_calls)
            
            # Save the best model
            if self.best_model_save_path and avg_metrics["sharpe_ratio"] > self.best_mean_sharpe:
                self.best_mean_sharpe = avg_metrics["sharpe_ratio"]
                self.logger.info(f"New best Sharpe ratio: {self.best_mean_sharpe:.4f}. Saving model...")
                self.model.save(self.best_model_save_path) 