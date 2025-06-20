"""Trading environment module.

This module provides the core trading environment
that replaces the old code with proper separation of concerns.
"""

from .config import FeeStructure, TradingEnvironmentConfig, load_trading_config
from .core import TradingEnv

__all__ = [
    "TradingEnv",
    "TradingEnvironmentConfig", 
    "FeeStructure",
    "load_trading_config",
] 