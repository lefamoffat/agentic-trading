"""
API client for Forex.com broker.
"""

import urllib.parse
from typing import Dict, Any, Optional, Tuple
import json

import aiohttp

from src.brokers.forex_com.auth import AuthenticationHandler
from src.utils.logger import get_logger
from src.brokers.exceptions import JsonParseError
from src.brokers.forex_com.types import ForexComApiResponseKeys


class ApiClient:
    """Handles HTTP requests to the GainCapital API."""

    API_BASE_URL = "https://ciapi.cityindex.com/TradingAPI"

    def __init__(self, auth_handler: AuthenticationHandler):
        """
        Initialize the API client.

        Args:
            auth_handler: The authentication handler instance.
        """
        self.auth_handler = auth_handler
        self.logger = get_logger(__name__)
        self._market_id_cache: Dict[str, str] = {}

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        log_endpoint: bool = True
    ) -> Tuple[int, Any]:
        """
        Make an authenticated HTTP request to the GainCapital API.

        Args:
            method: HTTP method ('GET' or 'POST').
            endpoint: API endpoint URL (path part).
            params: Query parameters for GET requests.
            json_data: JSON payload for POST requests.
            log_endpoint: Whether to log the endpoint being called.

        Returns:
            A tuple of (status_code, response_data).
        """
        if not self.auth_handler.is_authenticated:
            raise Exception("Cannot make API request without being authenticated.")

        full_url = f"{self.API_BASE_URL}{endpoint}"
        headers = self.auth_handler.get_headers()

        try:
            async with aiohttp.ClientSession() as session:
                if log_endpoint:
                    self.logger.info(f"Calling {method} endpoint: {full_url}")

                async with session.request(method.upper(), full_url, headers=headers, json=json_data, params=params) as response:
                    try:
                        # Attempt to parse JSON regardless of the content-type header.
                        # This is more resilient to API variations like 'application/json; charset=utf-8'
                        data = await response.json(content_type=None)
                        return response.status, data
                    except (json.JSONDecodeError, aiohttp.ContentTypeError) as e:
                        # If JSON decoding fails, it's likely not a JSON response.
                        # This is a critical error if the status code was a success.
                        raw_text = await response.text()
                        if 200 <= response.status < 300:
                            self.logger.error(
                                f"Failed to decode JSON from successful API response. Status: {response.status}, Body: {raw_text[:200]}..."
                            )
                            raise JsonParseError(
                                f"API returned non-JSON response for a successful request. Status: {response.status}"
                            ) from e
                        else:
                            # If it's an error status, returning the text is fine.
                            return response.status, raw_text

        except Exception as e:
            self.logger.error(f"Error making {method} request to {full_url}: {e}")
            raise

    async def get_market_id(self, symbol: str) -> str:
        """
        Get GainCapital market ID for a symbol using the search API, with caching.

        Args:
            symbol: Symbol in common format (e.g., "EUR/USD").

        Returns:
            The market ID string.

        Raises:
            ValueError: If no market is found for the symbol.
            Exception: If the market search API call fails.
        """
        if symbol in self._market_id_cache:
            return self._market_id_cache[symbol]

        try:
            encoded_symbol = urllib.parse.quote(symbol)
            params = {
                "SearchByMarketName": "TRUE",
                "Query": encoded_symbol,
                "MaxResults": "1"
            }
            endpoint = "/market/search"
            status, data = await self._make_request('GET', endpoint, params=params, log_endpoint=False)

            if status == 200:
                markets = data.get(ForexComApiResponseKeys.MARKETS, [])
                if markets:
                    market_id = str(markets[0][ForexComApiResponseKeys.MARKET_ID])
                    self._market_id_cache[symbol] = market_id
                    self.logger.debug(f"Found market ID {market_id} for symbol {symbol}")
                    return market_id
                else:
                    raise ValueError(f"No market found for symbol: {symbol}")
            else:
                raise Exception(f"Market search failed: {status} - {data}")

        except Exception as e:
            self.logger.error(f"Error getting market ID for {symbol}: {e}")
            raise 