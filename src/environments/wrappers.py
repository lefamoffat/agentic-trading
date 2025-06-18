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
        if done:
            # Copy trade history recorded so far
            self.trade_history = info.get("trade_history", [])

            # If an open position is still active, synthesise a closing trade so
            # metrics like `total_trades` and `profit_factor` are meaningful.
            env = self.env.unwrapped  # Original TradingEnv
            try:
                from src.environments.types import Position  # avoid top-level import

                if getattr(env, "_position", Position.FLAT) != Position.FLAT:
                    current_price = env.data.iloc[env.current_step]["close"]
                    entry_price = getattr(env, "_position_entry_price", current_price)
                    position = getattr(env, "_position")

                    profit = (
                        (current_price - entry_price)
                        if position == Position.LONG
                        else (entry_price - current_price)
                    ) * (env.balance / entry_price)

                    from src.environments.trading_env import Trade

                    synthetic_trade = Trade(
                        entry_price=entry_price,
                        exit_price=current_price,
                        position=position,
                        profit=profit,
                    )
                    self.trade_history.append(synthetic_trade)
            except Exception:  # pragma: no cover – safety; shouldn't break eval
                pass

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
