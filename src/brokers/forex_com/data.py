"""
Data handler for Forex.com broker.
"""

from typing import Dict, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd

from src.brokers.forex_com.api import ApiClient
from src.utils.logger import get_logger
from src.brokers.symbol_mapper import SymbolMapper
from src.types import BrokerType
from src.brokers.forex_com.types import ForexComApiResponseKeys, ForexComApiParams


class DataHandler:
    """Handles historical and live price data operations."""

    def __init__(self, api: ApiClient):
        """
        Initialize the data handler.

        Args:
            api: The API client instance.
        """
        self.api = api
        self.logger = get_logger(__name__)
        self.symbol_mapper = SymbolMapper(broker_type=BrokerType.FOREX_COM)

    def _get_timeframe_params(self, timeframe: str) -> Tuple[str, int]:
        """Maps our standard timeframes to GainCapital API intervals."""
        interval_map = {
            "1m": ("MINUTE", 1),
            "5m": ("MINUTE", 5),
            "15m": ("MINUTE", 15),
            "30m": ("MINUTE", 30),
            "1h": ("HOUR", 1),
            "4h": ("HOUR", 4),
            "1d": ("DAY", 1)
        }
        return interval_map.get(timeframe, ("HOUR", 1))

    def _parse_dotnet_date(self, date_str: str) -> datetime:
        """Parses Microsoft .NET JSON date format: /Date(timestamp)/."""
        if "/Date(" in date_str:
            timestamp_ms = int(date_str.split("(")[1].split(")")[0])
            return pd.to_datetime(timestamp_ms, unit='ms')
        else:
            return pd.to_datetime(date_str)

    async def get_historical_data(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        """
        Fetch historical price data for a given symbol and timeframe.
        """
        market_id = await self.api.get_market_id(symbol)
        interval, span = self._get_timeframe_params(timeframe)
        
        endpoint = f"/market/{market_id}/barhistory"
        
        # The Forex.com API uses 'interval' and a different naming scheme
        params = {
            ForexComApiParams.INTERVAL: interval,
            ForexComApiParams.SPAN: span,
            ForexComApiParams.PRICE_BARS: bars,
        }

        status, data = await self.api._make_request('GET', endpoint, params=params)
        
        # Process and format the data
        if status != 200 or not data or ForexComApiResponseKeys.PRICE_BARS not in data:
            raise Exception(f"API request failed: Status {status}, Response: {data}")

        price_bars = data[ForexComApiResponseKeys.PRICE_BARS]
        rows = [
            {
                "timestamp": self._parse_dotnet_date(bar["BarDate"]),
                "open": float(bar["Open"]),
                "high": float(bar["High"]),
                "low": float(bar["Low"]),
                "close": float(bar["Close"]),
                "volume": int(bar.get("Volume", 0))
            }
            for bar in price_bars
        ]
        return pd.DataFrame(rows)

    async def get_live_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get the current live price.

        Since the API doesn't provide a direct live price endpoint, this fetches
        the latest historical bar and market spread information.

        Args:
            symbol: The symbol in common format.

        Returns:
            A dictionary with bid, ask, mid, spread, and timestamp.
        """
        try:
            market_id = await self.api.get_market_id(symbol)

            # Get the latest bar to estimate current price
            hist_endpoint = f"/market/{market_id}/barhistory"
            hist_params = {
                ForexComApiParams.INTERVAL: "MINUTE",
                ForexComApiParams.SPAN: 5,
                ForexComApiParams.PRICE_BARS: 1
            }
            hist_status, hist_data = await self.api._make_request('GET', hist_endpoint, params=hist_params, log_endpoint=False)

            if hist_status != 200 or not hist_data.get(ForexComApiResponseKeys.PRICE_BARS):
                raise Exception(f"Could not get latest bar for live price: {hist_status} - {hist_data}")

            latest_bar = hist_data[ForexComApiResponseKeys.PRICE_BARS][0]
            close_price = float(latest_bar["Close"])
            timestamp = self._parse_dotnet_date(latest_bar["BarDate"])

            # Get market spread information
            info_endpoint = f"/market/{market_id}/information"
            info_status, info_data = await self.api._make_request('GET', info_endpoint, log_endpoint=False)

            spread = 0.0001  # Default spread
            if info_status == 200:
                market_info = info_data.get("MarketInformation", {})
                spreads = market_info.get("MarketSpreads", [])
                if spreads:
                    spread = spreads[0].get("Spread", spread)

            half_spread = spread / 2
            bid = close_price - half_spread
            ask = close_price + half_spread

            return {
                "symbol": symbol,
                "bid": bid,
                "ask": ask,
                "mid": close_price,
                "spread": spread,
                "timestamp": timestamp
            }
        except Exception as e:
            self.logger.error(f"Error getting live price for {symbol}: {e}")
            raise 