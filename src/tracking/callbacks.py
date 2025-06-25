#!/usr/bin/env python3
"""ML Tracking Callbacks for Stable-Baselines3.

Uses the generic tracking abstraction - works with Aim, Weights & Biases, etc.
"""

from typing import Optional, Dict, Any
import asyncio

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.tracking.utils import calculate_performance_metrics
from src.tracking import get_ml_tracker, TrainingMetrics

class TrackingCallback(BaseCallback):
    """A generic callback to log evaluation metrics to ML tracking backends.

    This callback runs policy evaluation at regular intervals and logs key
    performance metrics (Sharpe ratio, profit, drawdown) to the active
    tracking run using the generic interface.
    """

    def __init__(
        self,
        eval_env,
        run_id: str,
        eval_freq: int,
        n_eval_episodes: int = 5,
        timeframe: str = "1d",
        best_model_save_path: Optional[str] = None,
        verbose: int = 0
    ):
        super().__init__(verbose)
        # Use the environment directly - evaluation data now comes from info dict
        self.eval_env = eval_env
        self.run_id = run_id
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.timeframe = timeframe
        self.best_model_save_path = best_model_save_path
        self.best_mean_sharpe = -np.inf
        
        # Generic tracker instance
        self.tracker = None

    def _on_training_start(self) -> None:
        """Initialize the generic tracker."""
        try:
            # Get the generic tracker (works with any backend)
            self.tracker = asyncio.run(get_ml_tracker())
        except Exception as e:
            self.logger.warning(f"Failed to initialize tracking: {e}")
            self.tracker = None

    def _on_step(self) -> bool:
        """This method is called after each step in the training process."""
        if self.n_calls % self.eval_freq == 0:
            self.run_evaluation()
        return True

    def run_evaluation(self):
        """Run policy evaluation and log the results to the tracking backend."""
        if not self.tracker:
            # Skip if tracker not available
            return
        
        try:
            all_metrics = []
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                final_info = {}
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = self.eval_env.step(action)
                    final_info = info  # Keep the latest info
                    done = terminated or truncated

                # After the episode, calculate metrics from the environment's info
                # Get data from the last info dict (accumulated during episode)
                portfolio_values = [final_info.get('portfolio_value', 0)]  # Simplified for now
                trade_history = final_info.get('trade_history', [])
                metrics = calculate_performance_metrics(
                    portfolio_values,
                    trade_history,
                    self.timeframe
                )
                all_metrics.append(metrics)
        except Exception as e:
            if hasattr(self, 'model') and hasattr(self.model, 'logger'):
                self.model.logger.warning(f"Failed to run evaluation: {e}")
            return

        # Average the metrics over all evaluation episodes
        if all_metrics:
            avg_metrics = {
                key: np.mean([m[key] for m in all_metrics])
                for key in all_metrics[0]
            }

            # Log all metrics to the console
            for key, value in avg_metrics.items():
                self.logger.record(f"eval/{key}", value)

            # Create structured training metrics
            training_metrics = TrainingMetrics(
                reward=avg_metrics.get('total_return', 0.0),
                portfolio_value=avg_metrics.get('portfolio_value', 0.0),
                sharpe_ratio=avg_metrics.get('sharpe_ratio', 0.0),
                max_drawdown=avg_metrics.get('max_drawdown', 0.0),
                win_rate=avg_metrics.get('win_rate', 0.0),
                total_trades=int(avg_metrics.get('total_trades', 0)),
                avg_trade_return=avg_metrics.get('avg_trade_return', 0.0),
                profit_factor=avg_metrics.get('profit_factor', 0.0)
            )
            
            # Log to generic tracking backend
            try:
                asyncio.run(self.tracker.log_training_metrics(
                    self.run_id, 
                    training_metrics, 
                    self.n_calls
                ))
            except Exception as e:
                self.logger.warning(f"Failed to log metrics to tracking backend: {e}")

            # Save the best model
            if self.best_model_save_path and avg_metrics["sharpe_ratio"] > self.best_mean_sharpe:
                self.best_mean_sharpe = avg_metrics["sharpe_ratio"]
                self.logger.info(f"New best Sharpe ratio: {self.best_mean_sharpe:.4f}. Saving model...")
                self.model.save(self.best_model_save_path)

class SimpleTrackingCallback(BaseCallback):
    """A simple callback that logs training metrics without evaluation.
    
    This is useful for logging basic training progress without running
    separate evaluation episodes.
    """
    
    def __init__(
        self,
        run_id: str,
        log_freq: int = 1000,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.run_id = run_id
        self.log_freq = log_freq
        self.tracker = None
    
    def _on_training_start(self) -> None:
        """Initialize the generic tracker."""
        try:
            self.tracker = asyncio.run(get_ml_tracker())
        except Exception as e:
            self.logger.warning(f"Failed to initialize tracking: {e}")
            self.tracker = None
    
    def _on_step(self) -> bool:
        """Log basic training metrics."""
        if not self.tracker:
            return True
            
        if self.n_calls % self.log_freq == 0:
            # Get basic training info
            info_dict = self.locals.get('infos', [{}])[-1]  # Latest info
            
            # Create basic training metrics from available info
            training_metrics = TrainingMetrics(
                reward=float(self.locals.get('rewards', [0])[-1]),
                # Add more metrics as available from the training loop
            )
            
            # Log basic metrics
            try:
                asyncio.run(self.tracker.log_metrics(
                    self.run_id,
                    {"step_reward": training_metrics.reward},
                    self.n_calls
                ))
            except Exception as e:
                self.logger.warning(f"Failed to log basic metrics: {e}")
        
        return True 