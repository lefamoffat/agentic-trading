#!/usr/bin/env python3
"""Custom Gymnasium wrappers for the trading environment.
"""
from typing import Any, Dict, List, Tuple

import gymnasium as gym

from src.environments.trading_env import Trade


class EvaluationWrapper(gym.Wrapper):
    """A wrapper to collect portfolio values and trade history during an
    evaluation episode.

    This wrapper records the 'portfolio_value' from the info dictionary
    at each step and stores the history in `self.portfolio_values`.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.portfolio_values: List[float] = []
        self.trade_history: List[Trade] = []

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Step the environment and record the portfolio value.
        If the episode is done, also record the final trade history.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        if "portfolio_value" in info:
            self.portfolio_values.append(info["portfolio_value"])

        done = terminated or truncated
        if done and "trade_history" in info:
            self.trade_history = info["trade_history"]

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment and clear the collected data.
        """
        self.portfolio_values = []
        self.trade_history = []

        # Record the initial portfolio value
        initial_obs, info = self.env.reset(**kwargs)
        if "portfolio_value" in info:
            self.portfolio_values.append(info["portfolio_value"])
        return initial_obs, info
