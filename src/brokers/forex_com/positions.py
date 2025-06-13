"""
Position handler for Forex.com broker.
"""

from typing import List

from src.brokers.forex_com.api import ApiClient
from src.brokers.base import Position
from src.brokers.symbol_mapper import SymbolMapper
from src.utils.logger import get_logger


class PositionHandler:
    """Handles position-related operations."""

    def __init__(self, api_client: ApiClient, symbol_mapper: SymbolMapper):
        """
        Initialize the position handler.

        Args:
            api_client: The API client instance.
            symbol_mapper: The symbol mapper instance.
        """
        self.api_client = api_client
        self.symbol_mapper = symbol_mapper
        self.logger = get_logger(__name__)

    async def get_positions(self) -> List[Position]:
        """
        Get current positions using GainCapital API v2.

        Returns:
            A list of Position objects.

        Raises:
            Exception: If the API request fails.
        """
        try:
            endpoint = "/order/openpositions"
            status, data = await self.api_client._make_request('GET', endpoint)

            if status == 200:
                positions = []
                open_positions = data.get("OpenPositions", [])
                for pos_data in open_positions:
                    gc_symbol = pos_data.get("Market", {}).get("Name", "")
                    try:
                        common_symbol = self.symbol_mapper.from_broker_symbol(gc_symbol)
                    except ValueError:
                        self.logger.warning(f"Could not map broker symbol '{gc_symbol}' to common symbol. Skipping position.")
                        continue

                    quantity = float(pos_data.get("Quantity", 0))
                    price = float(pos_data.get("Price", 0))
                    pnl = float(pos_data.get("PnL", 0))

                    if quantity != 0:
                        position = Position(
                            symbol=common_symbol,
                            quantity=quantity,
                            avg_price=price,
                            unrealized_pnl=pnl
                        )
                        positions.append(position)
                return positions
            else:
                raise Exception(f"API request failed: {status} - {data}")

        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            raise
