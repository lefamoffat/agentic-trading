"""Main broker class for Forex.com integration.
"""

from typing import Dict, List, Optional

import pandas as pd

from src.brokers.base import (
    BaseBroker,
    Order,
    OrderStatus,
    Position,
)
from src.brokers.forex_com.account import AccountHandler
from src.brokers.forex_com.api import ApiClient
from src.brokers.forex_com.auth import AuthenticationHandler
from src.brokers.forex_com.data import DataHandler
from src.brokers.forex_com.orders import OrderHandler
from src.brokers.forex_com.positions import PositionHandler
from src.brokers.symbol_mapper import BrokerType as BrokerTypeEnum
from src.brokers.symbol_mapper import SymbolMapper
from src.utils.logger import get_logger

class ForexComBroker(BaseBroker):
    """Forex.com broker implementation using the composition pattern.
    This class orchestrates the different handlers for auth, data, orders, etc.
    """

    def __init__(self, api_key: str, api_secret: str, sandbox: bool = True):
        """Initialize the Forex.com broker.

        Args:
            api_key: The API key (username).
            api_secret: The API secret (password).
            sandbox: Whether to use the sandbox environment (ignored by this broker).

        """
        super().__init__(api_key, api_secret, sandbox)
        self.logger = get_logger(__name__)
        self.symbol_mapper = SymbolMapper(BrokerTypeEnum.FOREX_COM)

        # Composition: The broker "has" handlers for specific responsibilities.
        self.auth_handler = AuthenticationHandler(api_key, api_secret)
        self.api_client = ApiClient(self.auth_handler)
        self.account_handler = AccountHandler(self.api_client)
        self.position_handler = PositionHandler(self.api_client, self.symbol_mapper)
        self.order_handler = OrderHandler(self.api_client, self.symbol_mapper)
        self.data_handler = DataHandler(self.api_client)

    async def authenticate(self) -> bool:
        """Authenticate with the broker API by delegating to the auth handler."""
        return await self.auth_handler.authenticate()

    async def get_account_info(self) -> Dict:
        """Get account information by delegating to the account handler."""
        return await self.account_handler.get_account_info()

    async def get_positions(self) -> List[Position]:
        """Get current positions by delegating to the position handler."""
        return await self.position_handler.get_positions()

    async def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get orders by delegating to the order handler."""
        return await self.order_handler.get_orders(status)

    async def place_order(self, order: Order) -> str:
        """Place an order by delegating to the order handler."""
        return await self.order_handler.place_order(order)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by delegating to the order handler."""
        return await self.order_handler.cancel_order(order_id)

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        bars: int
    ) -> pd.DataFrame:
        """Get historical price data by delegating to the data handler."""
        return await self.data_handler.get_historical_data(symbol, timeframe, bars)

    async def get_live_price(self, symbol: str) -> Dict:
        """Get live price data by delegating to the data handler."""
        return await self.data_handler.get_live_price(symbol)

    def map_symbol_to_broker(self, common_symbol: str) -> str:
        """Map a common symbol to the broker-specific format."""
        return self.symbol_mapper.to_broker_symbol(common_symbol)

    def map_symbol_from_broker(self, broker_symbol: str) -> str:
        """Map a broker-specific symbol to the common format."""
        return self.symbol_mapper.from_broker_symbol(broker_symbol)

    async def get_all_positions(self) -> list[Position]:
        """Get all positions by delegating to the position handler.
        """
        return await self.position_handler.get_positions()
