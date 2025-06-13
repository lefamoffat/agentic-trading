#!/usr/bin/env python3
"""
Concrete implementation of the trading environment.
"""
from typing import Any, Dict

import numpy as np
import pandas as pd
from gymnasium import spaces

from src.environments.base import BaseTradingEnv
from src.environments.types import Position


class TradingEnv(BaseTradingEnv):
    """
    A concrete trading environment for reinforcement learning.

    This environment simulates trading a single asset. It provides a simplified
    action space (short, flat, long) and an observation space that includes
    market data and the current portfolio status.

    Attributes:
        action_space (spaces.Discrete): The action space (Short, Flat, Long).
        observation_space (spaces.Box): The observation space.
        trade_fee (float): The fee for executing a trade.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        trade_fee: float = 0.001,
    ):
        """
        Initialize the trading environment.

        Args:
            data (pd.DataFrame): DataFrame containing market data and features.
            initial_balance (float): The initial account balance.
            trade_fee (float): The transaction fee as a fraction of the trade amount.
        """
        super().__init__(data, initial_balance)
        self.trade_fee = trade_fee

        # Action space: 0 (Short), 1 (Flat), 2 (Long)
        self.action_space = spaces.Discrete(len(Position))

        # Observation space: [current_price, balance, current_position] + market features
        # The shape is 3 (for price, balance, position) + number of feature columns
        num_features = len(self.data.columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3 + num_features,), dtype=np.float32
        )

        # Portfolio state
        self._position = Position.FLAT
        self._position_entry_price = 0.0
        self._last_portfolio_value = self.initial_balance
        self._portfolio_value = self.initial_balance

    def _get_observation(self) -> np.ndarray:
        """
        Get the observation for the current step.

        The observation includes the current price, balance, position, and all
        other features from the input data frame.
        """
        features = self.data.iloc[self.current_step].values
        
        # Normalize balance to avoid large values dominating the observation
        normalized_balance = self.balance / self.initial_balance

        state = np.concatenate(
            (
                [
                    features[self.data.columns.get_loc("close")],
                    normalized_balance,
                    self._position.value,
                ],
                features,
            )
        ).astype(np.float32)

        return state

    def _get_info(self) -> Dict[str, Any]:
        """Get auxiliary information for the current step."""
        return {
            "step": self.current_step,
            "portfolio_value": self._portfolio_value,
            "balance": self.balance,
            "position": self._position.name,
            "entry_price": self._position_entry_price,
        }

    def _take_action(self, action: int) -> None:
        """Execute a trading action."""
        current_price = self.data.iloc[self.current_step]["close"]
        action = Position(action)

        if action == self._position:
            # No change in position
            return

        # Close current position before opening a new one
        if self._position != Position.FLAT:
            profit = 0
            if self._position == Position.LONG:
                profit = (current_price - self._position_entry_price) * (
                    self.balance / self._position_entry_price
                )
            elif self._position == Position.SHORT:
                profit = (self._position_entry_price - current_price) * (
                    self.balance / self._position_entry_price
                )
            
            self.balance += profit * (1 - self.trade_fee)
            self._position = Position.FLAT

        # Open new position
        if action != Position.FLAT:
            self._position = action
            self._position_entry_price = current_price

    def _calculate_reward(self) -> float:
        """
        Calculate the reward, defined as the change in portfolio value.
        """
        current_price = self.data.iloc[self.current_step]["close"]
        
        # Calculate current portfolio value
        if self._position == Position.LONG:
            unrealized_pnl = (current_price - self._position_entry_price) * (self.balance / self._position_entry_price)
            self._portfolio_value = self.balance + unrealized_pnl
        elif self._position == Position.SHORT:
            unrealized_pnl = (self._position_entry_price - current_price) * (self.balance / self._position_entry_price)
            self._portfolio_value = self.balance + unrealized_pnl
        else: # Position.FLAT
            self._portfolio_value = self.balance
        
        reward = self._portfolio_value - self._last_portfolio_value
        self._last_portfolio_value = self._portfolio_value
        
        return reward

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to its initial state."""
        super().reset(seed=seed)
        self._position = Position.FLAT
        self._position_entry_price = 0.0
        self._last_portfolio_value = self.initial_balance
        self._portfolio_value = self.initial_balance
        
        return self._get_observation(), self._get_info() 