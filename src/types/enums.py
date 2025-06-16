"""
Common, project-wide enums.
"""
from enum import Enum

# Broker-related enums
class BrokerType(Enum):
    """Supported broker types."""
    FOREX_COM = "forex_com"
    GENERIC = "generic"

class AssetClass(Enum):
    """Asset class classifications."""
    FOREX = "forex"
    STOCKS = "stocks"
    COMMODITIES = "commodities"
    CRYPTO = "crypto"
    INDICES = "indices"
    BONDS = "bonds"

class Timeframe(Enum):
    """Standard timeframes for market data."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"
    
    @property
    def qlib_name(self) -> str:
        """Get the Qlib-compatible name for the timeframe."""
        mapping = {
            "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
            "1h": "60min", "4h": "240min", "1d": "1d", "1w": "1w", "1M": "1mon"
        }
        return mapping[self.value]
    
    @property
    def minutes(self) -> int:
        """Convert timeframe to minutes."""
        mapping = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440, "1w": 10080, "1M": 43200
        }
        return mapping[self.value]
    
    @classmethod
    def from_minutes(cls, minutes: int) -> 'Timeframe':
        """Create timeframe from minutes."""
        reverse_mapping = {
            1: cls.M1, 5: cls.M5, 15: cls.M15, 30: cls.M30,
            60: cls.H1, 240: cls.H4, 1440: cls.D1, 10080: cls.W1, 43200: cls.MN1
        }
        if minutes not in reverse_mapping:
            raise ValueError(f"Unsupported timeframe: {minutes} minutes")
        return reverse_mapping[minutes]

    @classmethod
    def from_standard(cls, timeframe_str: str) -> 'Timeframe':
        """Create a Timeframe instance from a standard string like '1h' or '1d'."""
        for member in cls:
            if member.value == timeframe_str:
                return member
        raise ValueError(f"'{timeframe_str}' is not a valid Timeframe string.")

    @classmethod
    def get_display_list(cls) -> list[str]:
        """Returns a list of all timeframe string values."""
        return [item.value for item in cls]

# Trading-related enums
class OrderType(Enum):
    """Order types for trading."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status states."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PARTIALLY_FILLED = "partially_filled"

# Data processing types
class DataQuality(Enum):
    """Data quality indicators."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    INVALID = "invalid"

# Calendar and time types
class MarketSession(Enum):
    """Market trading session types."""
    CLOSED = "closed"
    OPEN = "open"
    PRE_MARKET = "pre_market"
    POST_MARKET = "post_market"
    OVERLAP = "overlap"  # For forex session overlaps

class EventImportance(Enum):
    """Economic event importance levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Strategy types
class StrategyType(Enum):
    """Trading strategy types."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    SCALPING = "scalping"
    SWING = "swing"
    POSITION = "position"

class SignalType(Enum):
    """Trading signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"

# Risk management types
class RiskLevel(Enum):
    """Risk level classifications."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high" 