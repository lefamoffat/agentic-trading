#!/usr/bin/env python3
"""Portfolio observation component for trading environment.

This module handles portfolio-related observations like balance, position, PnL.
"""
from typing import Optional

import numpy as np

from src.environment.state.position import Position, PositionManager
from src.environment.state.portfolio import PortfolioTracker


class PortfolioObservation:
    """Handles portfolio state observations.
    
    This provides clean portfolio state without legacy normalization issues.
    """
    
    def __init__(self, include_balance: bool = True, include_position: bool = True,
                 normalize_balance: bool = True):
        """Initialize portfolio observation component.
        
        Args:
            include_balance: Whether to include balance in observation
            include_position: Whether to include position state in observation  
            normalize_balance: Whether to normalize balance by initial balance
        """
        self.include_balance = include_balance
        self.include_position = include_position
        self.normalize_balance = normalize_balance
    
    def get_observation(self, position_manager: PositionManager, 
                       portfolio_tracker: PortfolioTracker,
                       current_price: float) -> np.ndarray:
        """Get portfolio state observation.
        
        Args:
            position_manager: Current position manager
            portfolio_tracker: Current portfolio tracker
            current_price: Current market price
            
        Returns:
            Portfolio state observation array
        """
        observations = []
        
        # Add balance information
        if self.include_balance:
            if self.normalize_balance:
                normalized_balance = portfolio_tracker.balance / portfolio_tracker.initial_balance
                observations.append(normalized_balance)
            else:
                observations.append(portfolio_tracker.balance)
        
        # Add position information
        if self.include_position:
            # Position type as float (0=SHORT, 1=FLAT, 2=LONG)
            observations.append(float(position_manager.position.value))
            
            # Position entry price (normalized by current price if position open)
            if not position_manager.is_flat and position_manager.entry_price is not None:
                normalized_entry = position_manager.entry_price / current_price
                observations.append(normalized_entry)
            else:
                observations.append(0.0)  # No position
            
            # Unrealized P&L as percentage of balance
            if not position_manager.is_flat and portfolio_tracker.balance > 0:
                position_size = portfolio_tracker.calculate_position_size(current_price)
                unrealized_pnl = position_manager.calculate_unrealized_pnl(current_price, position_size)
                pnl_pct = (unrealized_pnl / portfolio_tracker.balance) * 100
                observations.append(pnl_pct)
            else:
                observations.append(0.0)  # No unrealized P&L
        
        return np.array(observations, dtype=np.float32)
    
    def get_feature_names(self) -> list[str]:
        """Get list of feature names in observation order.
        
        Returns:
            List of feature names
        """
        features = []
        
        if self.include_balance:
            if self.normalize_balance:
                features.append("balance_normalized")
            else:
                features.append("balance")
        
        if self.include_position:
            features.extend(["position_type", "entry_price_normalized", "unrealized_pnl_pct"])
        
        return features
    
    @property
    def observation_size(self) -> int:
        """Get size of observation vector.
        
        Returns:
            Number of features in observation
        """
        size = 0
        if self.include_balance:
            size += 1
        if self.include_position:
            size += 3  # position_type, entry_price_normalized, unrealized_pnl_pct
        return size 