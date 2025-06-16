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
    A custom callback to log evaluation metrics to MLflow.
    
    This callback runs policy evaluation at regular intervals and logs key
    performance metrics (Sharpe ratio, profit, drawdown) to the active
    MLflow run.
    """

    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int = 5, verbose: int = 0):
        super().__init__(verbose)
        # It's recommended to wrap the eval env here for clarity
        self.eval_env = EvaluationWrapper(eval_env)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

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
            metrics = calculate_performance_metrics(portfolio_values)
            all_metrics.append(metrics)

        # Average the metrics over all evaluation episodes
        if all_metrics:
            avg_metrics = {
                key: np.mean([m[key] for m in all_metrics])
                for key in all_metrics[0]
            }
            
            self.logger.record("eval/sharpe_ratio", avg_metrics["sharpe_ratio"])
            self.logger.record("eval/profit_pct", avg_metrics["profit_pct"])
            self.logger.record("eval/max_drawdown_pct", avg_metrics["max_drawdown_pct"])
            
            mlflow.log_metrics(avg_metrics, step=self.n_calls) 