"""
Common, project-wide type aliases.
"""
from typing import Dict, List, Optional, Union, Any, Tuple, TypeAlias, Callable
import pandas as pd
from datetime import datetime

from .enums import OrderType, OrderSide, SignalType

# Core trading types
SymbolType: TypeAlias = str
PriceType: TypeAlias = float
VolumeType: TypeAlias = float
TimestampType: TypeAlias = Union[datetime, pd.Timestamp, str]

# Basic data structure types
OHLCVData: TypeAlias = pd.DataFrame
FeatureData: TypeAlias = pd.DataFrame
IndicatorData: TypeAlias = pd.DataFrame
SignalData: TypeAlias = pd.DataFrame
PriceData: TypeAlias = Dict[str, PriceType]
MarketData: TypeAlias = Dict[str, Any]
TradingSession: TypeAlias = Dict[str, Any]

# Configuration types
ConfigDict: TypeAlias = Dict[str, Any]
ParameterDict: TypeAlias = Dict[str, Union[int, float, str, bool]]

# Complex type aliases that depend on enums
AccountInfo: TypeAlias = Dict[str, Union[str, float, int]]
Position: TypeAlias = Dict[str, Union[str, float, datetime]]
Order: TypeAlias = Dict[str, Union[str, float, datetime, OrderType, OrderSide]]
TradeRecord: TypeAlias = Dict[str, Union[str, float, datetime, OrderType, OrderSide]]
PositionInfo: TypeAlias = Dict[str, Union[str, float, datetime]]

# Function signature types
DataValidator: TypeAlias = Callable[[pd.DataFrame], bool]
DataProcessor: TypeAlias = Callable[[pd.DataFrame], pd.DataFrame]
FeatureCalculator: TypeAlias = Callable[[pd.DataFrame], pd.DataFrame]
SignalGenerator: TypeAlias = Callable[[pd.DataFrame], SignalType]
ValidationFunction: TypeAlias = Callable[..., bool]
StrategyFunction: TypeAlias = Callable[[pd.DataFrame], pd.Series]

# Validation types
ValidationResult: TypeAlias = Dict[str, Union[bool, List[str]]]
ErrorContext: TypeAlias = Dict[str, Any]
ProcessingResult: TypeAlias = Tuple[bool, Optional[pd.DataFrame], Optional[str]] 