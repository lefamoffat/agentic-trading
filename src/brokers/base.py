"""
Base broker interface for trading system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import pandas as pd


class OrderType(Enum):
    """Order types supported by the trading system."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PARTIALLY_FILLED = "partially_filled"


class Position:
    """Represents a trading position."""
    
    def __init__(
        self,
        symbol: str,
        quantity: float,
        avg_price: float,
        unrealized_pnl: float = 0.0,
        realized_pnl: float = 0.0
    ):
        self.symbol = symbol
        self.quantity = quantity
        self.avg_price = avg_price
        self.unrealized_pnl = unrealized_pnl
        self.realized_pnl = realized_pnl
        
    @property
    def market_value(self) -> float:
        """Calculate market value of position."""
        return self.quantity * self.avg_price
        
    def __repr__(self) -> str:
        return f"Position(symbol={self.symbol}, qty={self.quantity}, avg_price={self.avg_price})"


class Order:
    """Represents a trading order."""
    
    def __init__(
        self,
        symbol: str,
        quantity: float,
        order_type: OrderType,
        side: OrderSide,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        order_id: Optional[str] = None,
        status: OrderStatus = OrderStatus.PENDING
    ):
        self.symbol = symbol
        self.quantity = quantity
        self.order_type = order_type
        self.side = side
        self.price = price
        self.stop_price = stop_price
        self.order_id = order_id
        self.status = status
        self.created_at = datetime.now()
        
    def __repr__(self) -> str:
        return f"Order(id={self.order_id}, symbol={self.symbol}, {self.side.value} {self.quantity} @ {self.price})"


class BaseBroker(ABC):
    """Abstract base class for broker integrations."""
    
    def __init__(self, api_key: str, api_secret: str, sandbox: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.sandbox = sandbox
        self._authenticated = False
        
    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with the broker API."""
        pass
        
    @abstractmethod
    async def get_account_info(self) -> Dict:
        """Get account information including balance and buying power."""
        pass
        
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        pass
        
    @abstractmethod
    async def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get orders, optionally filtered by status."""
        pass
        
    @abstractmethod
    async def place_order(self, order: Order) -> str:
        """Place an order and return order ID."""
        pass
        
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass
        
    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get historical price data."""
        pass
        
    @abstractmethod
    async def get_live_price(self, symbol: str) -> Dict:
        """Get current live price for a symbol."""
        pass
        
    @abstractmethod
    def map_symbol_to_broker(self, common_symbol: str) -> str:
        """Map common symbol format to broker-specific format."""
        pass
        
    @abstractmethod
    def map_symbol_from_broker(self, broker_symbol: str) -> str:
        """Map broker-specific symbol format to common format."""
        pass 