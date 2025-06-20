"""Trading environment module.

This module provides the core trading environment
that replaces the old code with proper separation of concerns.
"""

from src.environment.config import FeeStructure, TradingEnvironmentConfig, load_trading_config
from src.environment.core import TradingEnv

__all__ = [
    "TradingEnv",
    "TradingEnvironmentConfig", 
    "FeeStructure",
    "load_trading_config",
] 