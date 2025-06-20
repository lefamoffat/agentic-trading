#!/usr/bin/env python3
"""Composite reward system for combining multiple reward sources.

This module allows combining different reward calculations with configurable weights.
"""
from typing import Dict, List, Optional

from src.environment.reward_system.pnl_based import PnLBasedReward
from src.environment.state.position import Trade


class CompositeReward:
    """Combines multiple reward sources with configurable weights.
    
    This provides a flexible way to combine PnL rewards with risk penalties
    or other reward sources in the future.
    """
    
    def __init__(self, reward_components: Optional[Dict[str, float]] = None):
        """Initialize composite reward system.
        
        Args:
            reward_components: Dictionary mapping component names to weights
                             Default: {"pnl": 1.0} for pure PnL rewards
        """
        if reward_components is None:
            reward_components = {"pnl": 1.0}
        
        self.reward_components = reward_components
        self.component_calculators = {}
        
        # Initialize available reward calculators
        if "pnl" in reward_components:
            self.component_calculators["pnl"] = PnLBasedReward()
        
        # Future: add risk penalty calculators here
        # if "risk_penalty" in reward_components:
        #     self.component_calculators["risk_penalty"] = RiskPenaltyReward()
        
        self.reset()
    
    def reset(self) -> None:
        """Reset all reward component calculators."""
        for calculator in self.component_calculators.values():
            calculator.reset()
        
        self._last_reward = 0.0
        self._cumulative_reward = 0.0
        self._component_history: List[Dict[str, float]] = []
    
    def calculate_reward(self, trade: Optional[Trade] = None, portfolio_balance: float = 0.0, 
                        additional_data: Optional[Dict] = None) -> float:
        """Calculate composite reward from all components.
        
        Args:
            trade: Completed trade (if any) for this step
            portfolio_balance: Current portfolio balance
            additional_data: Additional data for reward calculations
            
        Returns:
            Weighted sum of all reward components
        """
        component_rewards = {}
        total_reward = 0.0
        
        # Calculate PnL reward
        if "pnl" in self.component_calculators:
            pnl_reward = self.component_calculators["pnl"].calculate_reward(trade, portfolio_balance)
            component_rewards["pnl"] = pnl_reward
            total_reward += pnl_reward * self.reward_components["pnl"]
        
        # Future: add other reward components
        # if "risk_penalty" in self.component_calculators:
        #     risk_reward = self.component_calculators["risk_penalty"].calculate_reward(...)
        #     component_rewards["risk_penalty"] = risk_reward
        #     total_reward += risk_reward * self.reward_components["risk_penalty"]
        
        # Store component breakdown for analysis
        self._component_history.append(component_rewards.copy())
        self._last_reward = total_reward
        self._cumulative_reward += total_reward
        
        return total_reward
    
    @property
    def last_reward(self) -> float:
        """Get the last calculated composite reward."""
        return self._last_reward
    
    @property
    def cumulative_reward(self) -> float:
        """Get the cumulative composite reward for the episode."""
        return self._cumulative_reward
    
    @property
    def component_history(self) -> List[Dict[str, float]]:
        """Get the history of component rewards for analysis."""
        return self._component_history.copy()
    
    def get_component_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each reward component.
        
        Returns:
            Dictionary with mean, std, min, max for each component
        """
        if not self._component_history:
            return {}
        
        stats = {}
        for component in self.reward_components.keys():
            values = [step.get(component, 0.0) for step in self._component_history]
            stats[component] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "total": sum(values)
            }
        
        return stats 