#!/usr/bin/env python3
"""Concrete implementation of the trading environment.
"""
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from gymnasium import spaces

from src.environments.base import BaseTradingEnv
from src.environments.types import Position


class Trade:
    """A data class to store details of a single trade."""

    def __init__(self, entry_price: float, exit_price: float, position: Position, profit: float):
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.position = position
        self.profit = profit


class TradingEnv(BaseTradingEnv):
    """Simplified trading environment – explicit CLOSE action and realised PnL reward.

    Action mapping (``spaces.Discrete(3)``)::

        0 – OPEN LONG
        1 – CLOSE (flat)
        2 – OPEN SHORT

    Reward::

        realised_PnL_pct * 100  (when a position is closed)  −  living_cost

    where ``living_cost`` defaults to ``0.01`` percentage-points per step.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10_000.0,
        trade_fee: float = 0.0,
        living_cost: float = 0.01,
    ) -> None:
        super().__init__(data, initial_balance)
        self.trade_fee = trade_fee
        self.living_cost = living_cost

        # 0 = OPEN LONG, 1 = CLOSE, 2 = OPEN SHORT
        self.action_space = spaces.Discrete(3)

        num_features = len(self.data.columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3 + num_features,), dtype=np.float32
        )

        # State
        self._position: Position = Position.FLAT
        self._position_entry_price: float = 0.0
        self.trade_history: List[Trade] = []

    @property
    def portfolio_value(self) -> float:
        """Calculate the current portfolio value.
        """
        current_price = self.data.iloc[self.current_step]["close"]

        # Calculate unrealized PnL
        unrealized_pnl = 0.0
        if self._position == Position.LONG:
            unrealized_pnl = (current_price - self._position_entry_price) * (self.balance / self._position_entry_price)
        elif self._position == Position.SHORT:
            unrealized_pnl = (self._position_entry_price - current_price) * (self.balance / self._position_entry_price)

        return self.balance + unrealized_pnl

    def _get_observation(self) -> np.ndarray:
        """Get the observation for the current step.

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
                    {Position.SHORT: -1.0, Position.FLAT: 0.0, Position.LONG: 1.0}[
                        self._position
                    ],
                ],
                features,
            )
        ).astype(np.float32)

        return state

    def _get_info(self) -> Dict[str, Any]:
        """Get auxiliary information for the current step."""
        return {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "balance": self.balance,
            "position": self._position.name,
            "entry_price": self._position_entry_price,
            "trade_history": self.trade_history,
        }

    # ------------------------------------------------------------------
    # Internal helpers required by BaseTradingEnv.step
    # ------------------------------------------------------------------

    def _take_action(self, action: int) -> None:  # type: ignore[override]
        """Execute the trading action and compute realised PnL if a position is closed."""
        price = self.data.iloc[self.current_step]["close"]
        self._realised_reward = 0.0  # reset

        if action == 0:  # OPEN LONG
            if self._position == Position.FLAT:
                self._position = Position.LONG
                self._position_entry_price = price

        elif action == 1:  # CLOSE
            if self._position != Position.FLAT:
                multiplier = 1.0 if self._position == Position.LONG else -1.0
                pnl = (price - self._position_entry_price) * multiplier * (
                    self.balance / self._position_entry_price
                )
                self._realised_reward = (pnl / self.balance) * 100 if self.balance else 0.0

                fee = self.balance * self.trade_fee
                self.balance = self.balance + pnl - fee

                self.trade_history.append(
                    Trade(
                        entry_price=self._position_entry_price,
                        exit_price=price,
                        position=self._position,
                        profit=pnl,
                    )
                )

                self._position = Position.FLAT
                self._position_entry_price = 0.0

        elif action == 2:  # OPEN SHORT
            if self._position == Position.FLAT:
                self._position = Position.SHORT
                self._position_entry_price = price
        else:
            raise ValueError("Invalid action index")

    def _calculate_reward(self) -> float:  # type: ignore[override]
        return self._realised_reward - self.living_cost

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to its initial state."""
        super().reset(seed=seed)
        self._position = Position.FLAT
        self._position_entry_price = 0.0
        self.balance = self.initial_balance
        self.trade_history = []
        # Initialize realised reward tracker
        self._realised_reward = 0.0
        return self._get_observation(), {}
