"""
Data handler for Forex.com broker.
"""

from typing import Dict, Tuple, Any
from datetime import datetime
import pandas as pd

from src.brokers.forex_com.api import ApiClient
from src.utils.logger import get_logger


class DataHandler:
    """Handles historical and live price data operations."""

    def __init__(self, api_client: ApiClient):
        """
        Initialize the data handler.

        Args:
            api_client: The API client instance.
        """
        self.api_client = api_client
        self.logger = get_logger(__name__)

    def _get_timeframe_params(self, timeframe: str) -> Tuple[str, int]:
        """Maps our standard timeframes to GainCapital API intervals."""
        interval_map = {
            "5m": ("MINUTE", 5),
            "15m": ("MINUTE", 15),
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

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        bars: int = 1000
    ) -> pd.DataFrame:
        """
        Get historical price data using GainCapital API v2.

        Args:
            symbol: The symbol in common format.
            timeframe: The timeframe (e.g., "5m", "1h").
            bars: The number of bars to fetch.

        Returns:
            A DataFrame with OHLCV data.
        """
        try:
            market_id = await self.api_client.get_market_id(symbol)
            interval, span = self._get_timeframe_params(timeframe)

            params = {
                "interval": interval,
                "span": span,
                "PriceBars": bars,
                "PriceType": "MID"
            }
            endpoint = f"/market/{market_id}/barhistory"
            status_code, data = await self.api_client._make_request('GET', endpoint, params=params)

            if status_code == 200:
                price_bars = data.get("PriceBars", [])
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
            else:
                raise Exception(f"API request failed: {status_code} - {data}")

        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            raise

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
            market_id = await self.api_client.get_market_id(symbol)

            # Get the latest bar to estimate current price
            hist_endpoint = f"/market/{market_id}/barhistory"
            hist_params = {"interval": "MINUTE", "span": 5, "PriceBars": 1, "PriceType": "MID"}
            hist_status, hist_data = await self.api_client._make_request('GET', hist_endpoint, params=hist_params, log_endpoint=False)

            if hist_status != 200 or not hist_data.get("PriceBars"):
                raise Exception(f"Could not get latest bar for live price: {hist_status} - {hist_data}")

            latest_bar = hist_data["PriceBars"][0]
            close_price = float(latest_bar["Close"])
            timestamp = self._parse_dotnet_date(latest_bar["BarDate"])

            # Get market spread information
            info_endpoint = f"/market/{market_id}/information"
            info_status, info_data = await self.api_client._make_request('GET', info_endpoint, log_endpoint=False)

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