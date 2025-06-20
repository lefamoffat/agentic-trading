#!/usr/bin/env python3
"""Composite observation component that combines multiple observation sources.

This module provides a unified observation interface for the trading environment.
"""
import numpy as np
import pandas as pd

from src.environment.observations.market import MarketObservation
from src.environment.observations.portfolio import PortfolioObservation
from src.environment.observations.time_features import TimeObservation
from src.environment.config import TradingEnvironmentConfig
from src.environment.state.position import PositionManager
from src.environment.state.portfolio import PortfolioTracker


class CompositeObservation:
    """Combines market, portfolio, and time observations into a unified observation.
    
    This provides a dynamic, extensible observation system that can handle:
    - Any market features (OHLCV, Qlib indicators, custom features)
    - Portfolio state (balance, position, P&L)
    - Time features (trading hours, market sessions, day of week)
    """
    
    def __init__(self, config: TradingEnvironmentConfig):
        """Initialize composite observation from configuration.
        
        Args:
            config: Complete trading environment configuration
        """
        self.config = config
        
        self.market_obs = MarketObservation(
            features=config.observation_features,
            normalization_method=config.normalization_method
        )
        
        self.portfolio_obs = PortfolioObservation(
            include_balance=config.include_portfolio_state,
            include_position=config.include_position_state
        )
        
        self.time_obs = TimeObservation(
            trading_start_hour=config.trading_start_hour,
            trading_end_hour=config.trading_end_hour,
            trading_timezone=config.trading_timezone,
            exclude_weekends=config.exclude_weekends
        ) if config.include_time_features else None
    
    def reset(self) -> None:
        """Reset all observation components."""
        self.market_obs.reset()
        # Portfolio obs doesn't need reset as it's stateless
        # Time obs doesn't need reset as it's stateless
    
    def get_observation(self, data: pd.DataFrame, current_step: int,
                       position_manager: PositionManager, 
                       portfolio_tracker: PortfolioTracker) -> np.ndarray:
        """Get unified observation combining market, portfolio, and time state.
        
        Args:
            data: Market data DataFrame (can contain any features)
            current_step: Current step index
            position_manager: Position manager instance
            portfolio_tracker: Portfolio tracker instance
            
        Returns:
            Combined observation array
        """
        observations = []
        
        # Get market observations (dynamic features)
        market_obs = self.market_obs.get_observation(data, current_step)
        observations.extend(market_obs)
        
        # Get portfolio observations if enabled
        if self.config.include_portfolio_state or self.config.include_position_state:
            current_price = data.iloc[current_step]["close"]
            portfolio_obs = self.portfolio_obs.get_observation(
                position_manager, portfolio_tracker, current_price
            )
            observations.extend(portfolio_obs)
        
        # Get time observations if enabled
        if self.config.include_time_features and self.time_obs is not None:
            time_obs = self.time_obs.get_observation(data, current_step)
            observations.extend(time_obs)
        
        return np.array(observations, dtype=np.float32)
    
    def get_feature_names(self) -> list[str]:
        """Get list of all feature names in observation order.
        
        Returns:
            List of feature names
        """
        features = []
        
        # Market features (dynamic)
        features.extend(self.market_obs.get_feature_names())
        
        # Portfolio features
        if self.config.include_portfolio_state or self.config.include_position_state:
            features.extend(self.portfolio_obs.get_feature_names())
        
        # Time features
        if self.config.include_time_features and self.time_obs is not None:
            features.extend(self.time_obs.get_feature_names())
        
        return features
    
    @property
    def observation_size(self) -> int:
        """Get total size of composite observation.
        
        Returns:
            Total number of features in observation
        """
        size = self.market_obs.observation_size
        
        if self.config.include_portfolio_state or self.config.include_position_state:
            size += self.portfolio_obs.observation_size
        
        if self.config.include_time_features and self.time_obs is not None:
            size += self.time_obs.observation_size
        
        return size
    
    def get_observation_space_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get reasonable bounds for observation space.
        
        Returns:
            Tuple of (low_bounds, high_bounds) arrays
        """
        # For normalized observations, reasonable bounds are:
        low_bounds = np.full(self.observation_size, -10.0, dtype=np.float32)
        high_bounds = np.full(self.observation_size, 10.0, dtype=np.float32)
        
        # Adjust bounds for specific feature types
        feature_names = self.get_feature_names()
        for i, feature_name in enumerate(feature_names):
            # Time features have specific bounds
            if feature_name in ["market_open", "time_of_day_normalized"] or feature_name.startswith("day_"):
                low_bounds[i] = 0.0
                high_bounds[i] = 1.0
            # Portfolio balance should be positive
            elif feature_name in ["balance_normalized"]:
                low_bounds[i] = 0.0
                high_bounds[i] = 10.0  # Allow for 10x growth
        
        return low_bounds, high_bounds 