#!/usr/bin/env python3
"""
Custom Gymnasium wrappers for the trading environment.
"""
from typing import List, Dict, Any, Tuple
import gymnasium as gym

class EvaluationWrapper(gym.Wrapper):
    """
    A wrapper to collect portfolio values during an evaluation episode.
    
    This wrapper records the 'portfolio_value' from the info dictionary
    at each step and stores the history in `self.portfolio_values`.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.portfolio_values: List[float] = []

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Step the environment and record the portfolio value.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        if "portfolio_value" in info:
            self.portfolio_values.append(info["portfolio_value"])
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment and clear the portfolio value history.
        """
        self.portfolio_values = []
        # Record the initial portfolio value
        initial_obs, info = self.env.reset(**kwargs)
        if "portfolio_value" in info:
            self.portfolio_values.append(info["portfolio_value"])
        return initial_obs, info 