"""
Forex.com broker integration for live trading.
"""

import asyncio
import aiohttp
import hashlib
import hmac
import time
import base64
import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import urllib.parse

from .base import BaseBroker, Order, Position, OrderType, OrderSide, OrderStatus
from .symbol_mapper import SymbolMapper, BrokerType
from ..utils.logger import get_logger


class ForexComBroker(BaseBroker):
    """
    Forex.com broker implementation using GainCapital/StoneX API v2.
    """
    
    # GainCapital API endpoints
    AUTH_BASE_URL = "https://ciapi.cityindex.com/v2"
    API_BASE_URL = "https://ciapi.cityindex.com/TradingAPI"
    
    def __init__(self, api_key: str, api_secret: str, sandbox: bool = True):
        """
        Initialize Forex.com broker.
        
        Args:
            api_key: GainCapital username
            api_secret: GainCapital password
            sandbox: Whether to use sandbox/demo environment (ignored - always uses live URL)
        """
        super().__init__(api_key, api_secret, sandbox)
        self.username = api_key
        self.password = api_secret
        self.app_key = os.getenv('FOREX_COM_APP_KEY')
        if not self.app_key:
            raise ValueError("FOREX_COM_APP_KEY environment variable is required")
            
        self.auth_base_url = self.AUTH_BASE_URL
        self.api_base_url = self.API_BASE_URL
        self.symbol_mapper = SymbolMapper(BrokerType.FOREX_COM)
        self.logger = get_logger(__name__)
        
        # GainCapital v2 specific session management
        self.session_token = None
        self.session_id = None
        self.user_account = None
        
        # Cache for market IDs to avoid repeated API calls
        self._market_id_cache = {}
        
    def _get_timeframe_params(self, timeframe: str) -> Tuple[str, int]:
        """
        Map timeframes to GainCapital intervals.
        
        Args:
            timeframe: Timeframe string (e.g., "5m", "1h", "1d")
            
        Returns:
            Tuple of (interval, span) for GainCapital API
        """
        interval_map = {
            "5m": ("MINUTE", 5),
            "15m": ("MINUTE", 15), 
            "1h": ("HOUR", 1),
            "4h": ("HOUR", 4),
            "1d": ("DAY", 1)
        }
        return interval_map.get(timeframe, ("HOUR", 1))
    
    def _parse_dotnet_date(self, date_str: str) -> datetime:
        """
        Parse Microsoft .NET JSON date format: /Date(timestamp)/.
        
        Args:
            date_str: Date string from GainCapital API
            
        Returns:
            Parsed datetime object
        """
        if "/Date(" in date_str:
            timestamp_ms = int(date_str.split("(")[1].split(")")[0])
            return pd.to_datetime(timestamp_ms, unit='ms')
        else:
            return pd.to_datetime(date_str)
    
    async def _ensure_authenticated(self):
        """Ensure we are authenticated before making API calls."""
        if not self._authenticated:
            await self.authenticate()
    
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
            method: HTTP method ('GET' or 'POST')
            endpoint: API endpoint URL
            params: Query parameters for GET requests
            json_data: JSON payload for POST requests
            log_endpoint: Whether to log the endpoint being called
            
        Returns:
            Tuple of (status_code, response_data)
        """
        await self._ensure_authenticated()
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = self._get_headers()
                
                if log_endpoint:
                    self.logger.info(f"Calling {method} endpoint: {endpoint}")
                
                if method.upper() == 'GET':
                    async with session.get(endpoint, headers=headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            return response.status, data
                        else:
                            error_text = await response.text()
                            return response.status, error_text
                            
                elif method.upper() == 'POST':
                    async with session.post(endpoint, headers=headers, json=json_data) as response:
                        if response.status == 200:
                            data = await response.json()
                            return response.status, data
                        else:
                            error_text = await response.text()
                            return response.status, error_text
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                    
        except Exception as e:
            self.logger.error(f"Error making {method} request to {endpoint}: {e}")
            raise

    async def authenticate(self) -> bool:
        """
        Authenticate with GainCapital API v2.
        
        Returns:
            True if authentication successful
        """
        try:
            async with aiohttp.ClientSession() as session:
                login_data = {
                    "UserName": self.username,
                    "Password": self.password,
                    "AppKey": self.app_key,
                    "AppVersion": "2.0",
                    "AppComments": "Agentic Trading System v2"
                }
                
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                
                auth_endpoint = f"{self.auth_base_url}/Session"
                self.logger.info(f"Authenticating to endpoint: {auth_endpoint}")
                self.logger.debug(f"Login data: {dict((k, v if k != 'Password' else '***') for k, v in login_data.items())}")
                
                async with session.post(
                    auth_endpoint,
                    json=login_data,
                    headers=headers
                ) as response:
                    response_text = await response.text()
                    self.logger.debug(f"Authentication response status: {response.status}")
                    self.logger.debug(f"Authentication response: {response_text}")
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # Check if authentication was actually successful
                        status_code = data.get("statusCode", -1)
                        if status_code == 0:  # 0 indicates success in GainCapital API
                            self.session_token = data.get("session")  # lowercase 'session'
                            self.session_id = data.get("session")     # Use same value for both
                            
                            # Store user account data if available
                            if "UserAccount" in data:
                                self.user_account = data["UserAccount"]
                            
                            if self.session_token:
                                self._authenticated = True
                                self.logger.info(f"Successfully authenticated with Forex.com. Session: {self.session_token[:8]}...")
                                return True
                        else:
                            self.logger.error(f"Authentication failed with statusCode: {status_code}")
                            return False
                    
                    self.logger.error(f"Authentication failed: {response.status} - {response_text}")
                    return False
                        
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return False
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with session token for authenticated requests."""
        if not self.session_token:
            raise Exception("Not authenticated - call authenticate() first")
            
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Session": self.session_token,
            "UserName": self.username
        }
    
    async def _get_market_id(self, symbol: str) -> str:
        """
        Get GainCapital market ID for symbol using search API with caching.
        
        Args:
            symbol: Symbol in common format (e.g., "EUR/USD")
            
        Returns:
            Market ID string
        """
        # Check cache first
        if symbol in self._market_id_cache:
            return self._market_id_cache[symbol]
        
        try:
            # URL encode the symbol for the query
            encoded_symbol = urllib.parse.quote(symbol)
            
            params = {
                "SearchByMarketName": "TRUE",
                "Query": encoded_symbol,
                "MaxResults": "1"
            }
            
            endpoint = f"{self.api_base_url}/market/search"
            status, data = await self._make_request('GET', endpoint, params=params, log_endpoint=False)
            
            if status == 200:
                markets = data.get("Markets", [])
                
                if markets:
                    market_id = str(markets[0]["MarketId"])
                    # Cache the result
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
    
    async def get_account_info(self) -> Dict:
        """
        Get account information using GainCapital API v2.
        
        Returns:
            Dictionary with account info
        """
        try:
            endpoint = f"{self.api_base_url}/useraccount/ClientAndTradingAccount"
            status, data = await self._make_request('GET', endpoint)
            
            if status == 200:
                trading_accounts = data.get("TradingAccounts", [])
                if trading_accounts:
                    account = trading_accounts[0]
                    
                    return {
                        "account_id": account.get("TradingAccountId"),
                        "balance": float(account.get("AccountBalance", 0)),
                        "currency": account.get("AccountCurrency"),
                        "available_funds": float(account.get("AvailableFunds", 0)),
                        "margin_requirement": float(account.get("MarginRequirement", 0)),
                        "unrealized_pnl": float(account.get("UnrealizedPnL", 0)),
                        "account_name": account.get("AccountName"),
                        "account_status": account.get("AccountStatus")
                    }
            else:
                raise Exception(f"API request failed: {status} - {data}")
                        
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            raise
    
    async def get_positions(self) -> List[Position]:
        """
        Get current positions using GainCapital API v2.
        
        Returns:
            List of Position objects
        """
        try:
            endpoint = f"{self.api_base_url}/order/openpositions"
            status, data = await self._make_request('GET', endpoint)
            
            if status == 200:
                positions = []
                
                open_positions = data.get("OpenPositions", [])
                for pos_data in open_positions:
                    gc_symbol = pos_data.get("Market", {}).get("Name", "")
                    try:
                        common_symbol = self.symbol_mapper.from_broker_symbol(gc_symbol)
                    except ValueError:
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
    
    async def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """
        Get orders using GainCapital API v2.
        
        Args:
            status: Filter by order status
            
        Returns:
            List of Order objects
        """
        try:
            endpoint = f"{self.api_base_url}/order/activeorders"
            status_code, data = await self._make_request('GET', endpoint)
            
            if status_code == 200:
                orders = []
                
                active_orders = data.get("ActiveOrders", [])
                for order_data in active_orders:
                    gc_symbol = order_data.get("Market", {}).get("Name", "")
                    try:
                        common_symbol = self.symbol_mapper.from_broker_symbol(gc_symbol)
                    except ValueError:
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
                        status=OrderStatus.PENDING
                    )
                    orders.append(order)
                
                return orders
            else:
                raise Exception(f"API request failed: {status_code} - {data}")
                        
        except Exception as e:
            self.logger.error(f"Error getting orders: {e}")
            raise
    
    async def place_order(self, order: Order) -> str:
        """
        Place an order using GainCapital API v2.
        
        Args:
            order: Order object with order details
            
        Returns:
            Order ID string
        """
        try:
            # Get market ID for symbol
            market_id = await self._get_market_id(order.symbol)
            
            # Convert quantity based on side
            quantity = order.quantity if order.side == OrderSide.BUY else -order.quantity
            
            if order.order_type == OrderType.MARKET:
                order_request = {
                    "MarketId": int(market_id),
                    "Currency": "USD",
                    "AutoRollover": False,
                    "Direction": "buy" if order.side == OrderSide.BUY else "sell",
                    "Quantity": abs(order.quantity),
                    "TradingAccountId": self.user_account.get("TradingAccountId") if self.user_account else 0,
                    "BidPrice": order.price or 0,
                    "OfferPrice": order.price or 0,
                    "AuditId": str(int(time.time()))
                }
                endpoint = f"{self.api_base_url}/order/newtradeorder"
            else:
                endpoint = f"{self.api_base_url}/order/newstoplimitorder"
                # Stop/Limit order implementation would go here
                raise NotImplementedError("Stop/Limit orders not yet implemented")
            
            status_code, data = await self._make_request('POST', endpoint, json_data=order_request)
            
            if status_code == 200:
                order_id = str(data.get("OrderId", ""))
                self.logger.info(f"Order placed successfully: {order_id}")
                return order_id
            else:
                raise Exception(f"Order placement failed: {status_code} - {data}")
                        
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order using GainCapital API v2.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if cancellation successful
        """
        try:
            cancel_request = {
                "OrderId": int(order_id),
                "TradingAccountId": self.user_account.get("TradingAccountId") if self.user_account else 0
            }
            
            endpoint = f"{self.api_base_url}/order/cancel"
            status_code, data = await self._make_request('POST', endpoint, json_data=cancel_request)
            
            if status_code == 200:
                self.logger.info(f"Order cancelled successfully: {order_id}")
                return True
            else:
                self.logger.error(f"Order cancellation failed: {status_code} - {data}")
                return False
                        
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False
    
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        bars: int = 1000
    ) -> pd.DataFrame:
        """
        Get historical price data using GainCapital API v2.
        
        Args:
            symbol: Symbol in common format
            timeframe: Timeframe (e.g., "5m", "1h", "1d")
            bars: Number of bars to fetch (default: 1000)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Get market ID for symbol
            market_id = await self._get_market_id(symbol)
            
            # Get timeframe parameters
            interval, span = self._get_timeframe_params(timeframe)
            
            params = {
                "interval": interval,
                "span": span,
                "PriceBars": bars,
                "PriceType": "MID"
            }
            
            endpoint = f"{self.api_base_url}/market/{market_id}/barhistory"
            
            # Debug headers being sent
            self.logger.info(f"Headers being sent: {self._get_headers()}")
            
            status_code, data = await self._make_request('GET', endpoint, params=params)
            
            self.logger.info(f"Historical data response status: {status_code}")
            
            if status_code == 200:
                price_bars = data.get("PriceBars", [])
                
                rows = []
                for bar in price_bars:
                    # Parse Microsoft .NET JSON date format
                    timestamp = self._parse_dotnet_date(bar["BarDate"])
                    
                    row = {
                        "timestamp": timestamp,
                        "open": float(bar["Open"]),
                        "high": float(bar["High"]),
                        "low": float(bar["Low"]),
                        "close": float(bar["Close"]),
                        "volume": int(bar.get("Volume", 0))
                    }
                    rows.append(row)
                
                df = pd.DataFrame(rows)
                # Keep timestamp as column for standardization
                return df
            else:
                raise Exception(f"API request failed: {status_code} - {data}")
                        
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            raise
    
    async def get_live_price(self, symbol: str) -> Dict:
        """
        Get current live price using GainCapital API v2.
        
        Since the /information endpoint doesn't provide live prices,
        we'll get the latest bar from historical data.
        
        Args:
            symbol: Symbol in common format
            
        Returns:
            Dictionary with bid, ask, spread info
        """
        try:
            # Get the latest 1 bar from historical data to get current price
            market_id = await self._get_market_id(symbol)
            
            # Use shortest timeframe for most recent data
            interval, span = self._get_timeframe_params("5m")
            
            params = {
                "interval": interval,
                "span": span,
                "PriceBars": 1,  # Just get the latest bar
                "PriceType": "MID"
            }
            
            endpoint = f"{self.api_base_url}/market/{market_id}/barhistory"
            status_code, data = await self._make_request('GET', endpoint, params=params)
            
            if status_code == 200:
                price_bars = data.get("PriceBars", [])
                
                if price_bars:
                    latest_bar = price_bars[0]  # Get the most recent bar
                    
                    # Use close price as mid, estimate bid/ask with spread
                    close_price = float(latest_bar["Close"])
                    
                    # Get spread from market information if available
                    info_endpoint = f"{self.api_base_url}/market/{market_id}/information" 
                    info_status, info_data = await self._make_request('GET', info_endpoint, log_endpoint=False)
                    
                    # Default spread (in pips converted to price units)
                    default_spread = 0.0001  # 1 pip for EUR/USD
                    
                    if info_status == 200:
                        market_info = info_data.get("MarketInformation", {})
                        spreads = market_info.get("MarketSpreads", [])
                        if spreads:
                            spread = spreads[0].get("Spread", default_spread)
                        else:
                            spread = default_spread
                    else:
                        spread = default_spread
                    
                    # Calculate bid/ask from mid price and spread
                    half_spread = spread / 2
                    bid = close_price - half_spread
                    ask = close_price + half_spread
                    
                    self.logger.info(f"Live price for {symbol}: close={close_price}, spread={spread}, bid={bid}, ask={ask}")
                    
                    return {
                        "symbol": symbol,
                        "bid": bid,
                        "ask": ask,
                        "mid": close_price,
                        "spread": spread,
                        "timestamp": self._parse_dotnet_date(latest_bar["BarDate"])
                    }
                else:
                    raise Exception("No price bars returned from API")
            else:
                raise Exception(f"API request failed: {status_code} - {data}")
                        
        except Exception as e:
            self.logger.error(f"Error getting live price: {e}")
            raise
    
    def map_symbol_to_broker(self, common_symbol: str) -> str:
        """Map common symbol to GainCapital format."""
        return self.symbol_mapper.to_broker_symbol(common_symbol)
    
    def map_symbol_from_broker(self, broker_symbol: str) -> str:
        """Map GainCapital symbol to common format."""
        return self.symbol_mapper.from_broker_symbol(broker_symbol) 