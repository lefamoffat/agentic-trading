"""Order handler for Forex.com broker.
"""

import time
from typing import List, Optional

from src.brokers.base import Order, OrderSide, OrderStatus, OrderType
from src.brokers.forex_com.api import ApiClient
from src.brokers.forex_com.types import ForexComApiResponseKeys
from src.brokers.symbol_mapper import SymbolMapper
from src.utils.logger import get_logger


class OrderHandler:
    """Handles order-related operations."""

    def __init__(self, api_client: ApiClient, symbol_mapper: SymbolMapper):
        """Initialize the order handler.

        Args:
            api_client: The API client instance.
            symbol_mapper: The symbol mapper instance.

        """
        self.api_client = api_client
        self.symbol_mapper = symbol_mapper
        self.logger = get_logger(__name__)

    async def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get active orders using GainCapital API v2.

        Note: The API endpoint seems to only return active orders, so the status filter is ignored.

        Args:
            status: Filter by order status (currently ignored).

        Returns:
            A list of Order objects.

        """
        try:
            endpoint = "/order/activeorders"
            status_code, data = await self.api_client._make_request('GET', endpoint)

            if status_code == 200:
                orders = []
                active_orders = data.get(ForexComApiResponseKeys.ACTIVE_ORDERS, [])
                for order_data in active_orders:
                    gc_symbol = order_data.get("Market", {}).get("Name", "")
                    try:
                        common_symbol = self.symbol_mapper.from_broker_symbol(gc_symbol)
                    except ValueError:
                        self.logger.warning(f"Could not map broker symbol '{gc_symbol}' to common symbol. Skipping order.")
                        continue

                    quantity = abs(float(order_data.get("Quantity", 0)))
                    side = OrderSide.BUY if float(order_data.get("Quantity", 0)) > 0 else OrderSide.SELL

                    gc_type = order_data.get("Type", "")
                    if "Market" in gc_type:
                        order_type = OrderType.MARKET
                    elif "Limit" in gc_type:
                        order_type = OrderType.LIMIT
                    elif "Stop" in gc_type:
                        order_type = OrderType.STOP
                    else:
                        order_type = OrderType.MARKET

                    order = Order(
                        symbol=common_symbol,
                        quantity=quantity,
                        order_type=order_type,
                        side=side,
                        price=float(order_data.get("TriggerPrice", 0)),
                        order_id=str(order_data.get("OrderId")),
                        status=OrderStatus.PENDING  # API returns only active orders
                    )
                    orders.append(order)
                return orders
            else:
                raise Exception(f"API request failed: {status_code} - {data}")

        except Exception as e:
            self.logger.error(f"Error getting orders: {e}")
            raise

    async def place_order(self, order: Order) -> str:
        """Place an order using GainCapital API v2.

        Args:
            order: An Order object with order details.

        Returns:
            The order ID string.

        """
        try:
            market_id = await self.api_client.get_market_id(order.symbol)

            user_account_id = self.api_client.auth_handler.user_account.get("TradingAccountId") if self.api_client.auth_handler.user_account else 0

            if order.order_type == OrderType.MARKET:
                order_request = {
                    "MarketId": int(market_id),
                    "Currency": "USD",
                    "AutoRollover": False,
                    "Direction": "buy" if order.side == OrderSide.BUY else "sell",
                    "Quantity": abs(order.quantity),
                    "TradingAccountId": user_account_id,
                    "BidPrice": order.price or 0,
                    "OfferPrice": order.price or 0,
                    "AuditId": str(int(time.time()))
                }
                endpoint = "/order/newtradeorder"
            else:
                # The original implementation did not support this.
                raise NotImplementedError("Stop/Limit orders not yet implemented for Forex.com")

            status_code, data = await self.api_client._make_request('POST', endpoint, json_data=order_request)

            if status_code == 200:
                order_id = str(data.get("OrderId", ""))
                self.logger.info(f"Order placed successfully: {order_id}")
                return order_id
            else:
                raise Exception(f"Order placement failed: {status_code} - {data}")

        except Exception as e:
            self.logger.error(f"Error placing order for {order.symbol}: {e}")
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order using GainCapital API v2.

        Args:
            order_id: The ID of the order to cancel.

        Returns:
            True if cancellation was successful, False otherwise.

        """
        try:
            user_account_id = self.api_client.auth_handler.user_account.get("TradingAccountId") if self.api_client.auth_handler.user_account else 0
            cancel_request = {
                "OrderId": int(order_id),
                "TradingAccountId": user_account_id
            }

            endpoint = "/order/cancel"
            status_code, data = await self.api_client._make_request('POST', endpoint, json_data=cancel_request)

            if status_code == 200:
                self.logger.info(f"Order cancelled successfully: {order_id}")
                return True
            else:
                self.logger.error(f"Order cancellation failed: {status_code} - {data}")
                return False

        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
