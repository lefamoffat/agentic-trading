#!/usr/bin/env python3
"""Trading environment configuration.

This module provides type-safe configuration.
"""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

__all__ = [
    "ActionType",
    "FeeStructure", 
    "RewardSystem",
    "TradingEnvironmentConfig",
    "load_trading_config"
]


class ActionType(Enum):
    """Available action types for the trading environment."""
    DISCRETE_THREE = "discrete_3"


class FeeStructure(Enum):
    """Fee structure types."""
    SPREAD_BASED = "spread"        # Forex: bid/ask spread
    COMMISSION = "commission"      # Stocks: flat or percentage
    COMBINED = "combined"          # Both spread + commission


class RewardSystem(Enum):
    """Reward calculation systems."""
    PNL_BASED = "pnl"             # Profit/loss based rewards
    REALIZED_PNL = "realized_pnl"  # Realized profit/loss based rewards


@dataclass
class TradingEnvironmentConfig:
    """Configuration for the trading environment."""
    
    # Core settings
    initial_balance: float = 100000.0
    max_steps: Optional[int] = None
    
    # Fee structure
    fee_structure: FeeStructure = FeeStructure.SPREAD_BASED
    spread: float = 0.0001         # For forex (in price units)
    commission_rate: float = 0.0   # For stocks (percentage)
    
    pip_value: float = 0.0001      # Value of 1 pip (for forex)
    
    # Action configuration
    action_type: ActionType = ActionType.DISCRETE_THREE
    
    # Observation configuration - ALL fields needed by CompositeObservation
    observation_features: List[str] = None
    include_time_features: bool = True
    include_portfolio_state: bool = True
    include_position_state: bool = True
    normalization_method: str = "robust_zscore"
    
    # Position management - ALL fields needed by PortfolioTracker
    position_sizing_method: str = "fixed"
    
    # Reward system
    reward_system: RewardSystem = RewardSystem.REALIZED_PNL
    
    # Risk management
    max_drawdown: float = 0.20
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_daily_loss: Optional[float] = None
    max_risk_per_trade: Optional[float] = None
    
    # Trading hours configuration - ALL fields needed by TimeObservation
    trading_start_hour: int = 0
    trading_end_hour: int = 24
    trading_timezone: str = "UTC"
    exclude_weekends: bool = False
    exclude_holidays: bool = False
    
    # Live trading controls
    max_orders_per_day: Optional[int] = None
    cooldown_period: Optional[int] = None
    emergency_stop_loss: Optional[float] = None
    
    def __post_init__(self):
        """Set default observation features if not provided."""
        if self.observation_features is None:
            self.observation_features = ["close", "volume"]


def load_trading_config(config_path: Path) -> TradingEnvironmentConfig:
    """Load trading configuration from specified path.
    
    Args:
        config_path: Path to the trading_config.yaml file
        
    Returns:
        TradingEnvironmentConfig object with validated settings
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Extract relevant fields from trading_config.yaml
    backtesting = config_data.get('backtesting', {})
    broker = config_data.get('broker', {})
    actions = config_data.get('actions', {})
    instrument = config_data.get('instrument', {})
    position = config_data.get('position', {})
    risk = config_data.get('risk', {})
    trading_hours = config_data.get('trading_hours', {})
    live_trading = config_data.get('live_trading', {})
    
    # Parse trading hours with correct field names
    start_hour = trading_hours.get('trading_start_hour', 7)
    end_hour = trading_hours.get('trading_end_hour', 17)
    
    # Map to complete configuration with ALL required fields
    return TradingEnvironmentConfig(
        # Core settings from backtesting section
        initial_balance=float(backtesting.get('initial_cash', 100000)),
        
        # Fee structure - use spread for forex
        fee_structure=FeeStructure.SPREAD_BASED,
        spread=float(instrument.get('spread', 0.0001)),
        commission_rate=float(broker.get('commission', 0.0)),
        pip_value=float(instrument.get('pip_value', 0.0001)),
        
        # Action space based on config
        action_type=ActionType.DISCRETE_THREE,
        
        # Observation configuration - provide defaults for ALL fields
        observation_features=["close", "volume"],  # Default features
        include_time_features=True,
        include_portfolio_state=True,
        include_position_state=True,
        normalization_method="robust_zscore",
        
        # Position management
        position_sizing_method=position.get('sizing_method', 'fixed'),
        
        # Reward system
        reward_system=RewardSystem.REALIZED_PNL,
        
        # Risk management
        max_drawdown=float(risk.get('max_drawdown', 0.10)),
        stop_loss=risk.get('stop_loss'),
        take_profit=risk.get('take_profit'),
        max_daily_loss=risk.get('max_daily_loss'),
        max_risk_per_trade=risk.get('max_risk_per_trade'),
        
        # Trading hours - ALL fields needed by TimeObservation
        trading_start_hour=start_hour,
        trading_end_hour=end_hour,
        trading_timezone=trading_hours.get('trading_timezone', 'UTC'),
        exclude_weekends=trading_hours.get('exclude_weekends', True),
        exclude_holidays=trading_hours.get('exclude_holidays', True),
        
        # Trading frequency controls
        max_orders_per_day=live_trading.get('max_orders_per_day'),
        cooldown_period=live_trading.get('cooldown_period'),
        emergency_stop_loss=live_trading.get('emergency_stop_loss'),
    ) 