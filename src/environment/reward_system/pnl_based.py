#!/usr/bin/env python3
"""PnL-based reward system for trading environment.

This module implements clean profit/loss based rewards.
"""
from typing import Optional

from src.environment.state.position import Trade


class PnLBasedReward:
    """Calculates rewards based on realized profit and loss.
    
    This replaces the legacy reward system that included arbitrary 'living_cost'.
    """
    
    def __init__(self, scale_factor: float = 100.0):
        """Initialize PnL-based reward calculator.
        
        Args:
            scale_factor: Multiplier for reward scaling (default 100 for percentage)
        """
        self.scale_factor = scale_factor
        self.reset()
    
    def reset(self) -> None:
        """Reset reward tracking."""
        self._last_reward = 0.0
        self._cumulative_reward = 0.0
    
    def calculate_reward(self, trade: Optional[Trade] = None, portfolio_balance: float = 0.0) -> float:
        """Calculate reward for the current step.
        
        Args:
            trade: Completed trade (if any) for this step
            portfolio_balance: Current portfolio balance
            
        Returns:
            Reward value for this step
        """
        if trade is None:
            # No trade completed - no reward
            reward = 0.0
        else:
            # Reward based on trade profit as percentage of balance
            if portfolio_balance > 0:
                reward_pct = (trade.profit / portfolio_balance) * 100
                reward = reward_pct * (self.scale_factor / 100)
            else:
                reward = 0.0
        
        self._last_reward = reward
        self._cumulative_reward += reward
        return reward
    
    @property
    def last_reward(self) -> float:
        """Get the last calculated reward."""
        return self._last_reward
    
    @property
    def cumulative_reward(self) -> float:
        """Get the cumulative reward for the episode."""
        return self._cumulative_reward 