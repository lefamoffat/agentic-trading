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
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd

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
    
    def _get_market_id(self, symbol: str) -> str:
        """Get GainCapital market ID for symbol."""
        market_ids = {
            "EUR/USD": "402044081",
            "EURUSD": "402044081"
        }
        
        if symbol not in market_ids:
            raise ValueError(f"Unsupported symbol: {symbol}")
            
        return market_ids[symbol]
    
    async def get_account_info(self) -> Dict:
        """
        Get account information using GainCapital API v2.
        
        Returns:
            Dictionary with account info
        """
        if not self._authenticated:
            await self.authenticate()
            
        try:
            async with aiohttp.ClientSession() as session:
                headers = self._get_headers()
                
                endpoint = f"{self.api_base_url}/useraccount/ClientAndTradingAccount"
                self.logger.info(f"Calling account info endpoint: {endpoint}")
                
                async with session.get(
                    endpoint,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
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
                        raise Exception(f"API request failed: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            raise
    
    async def get_positions(self) -> List[Position]:
        """
        Get current positions using GainCapital API v2.
        
        Returns:
            List of Position objects
        """
        if not self._authenticated:
            await self.authenticate()
            
        try:
            async with aiohttp.ClientSession() as session:
                headers = self._get_headers()
                
                endpoint = f"{self.api_base_url}/order/openpositions"
                self.logger.info(f"Calling positions endpoint: {endpoint}")
                
                async with session.get(
                    endpoint,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
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
                        raise Exception(f"API request failed: {response.status}")
                        
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
        if not self._authenticated:
            await self.authenticate()
            
        try:
            async with aiohttp.ClientSession() as session:
                headers = self._get_headers()
                
                endpoint = f"{self.api_base_url}/order/activeorders"
                self.logger.info(f"Calling orders endpoint: {endpoint}")
                
                async with session.get(
                    endpoint,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
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
                            
                            if status is None or order.status == status:
                                orders.append(order)
                        
                        return orders
                    else:
                        raise Exception(f"API request failed: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Error getting orders: {e}")
            raise
    
    async def place_order(self, order: Order) -> str:
        """
        Place an order using GainCapital API v2.
        
        Args:
            order: Order object to place
            
        Returns:
            Order ID
        """
        if not self._authenticated:
            await self.authenticate()
            
        try:
            gc_symbol = self.symbol_mapper.to_broker_symbol(order.symbol)
            
            order_request = {
                "MarketId": gc_symbol,
                "Currency": "USD",
                "AutoRollover": False,
                "Direction": "buy" if order.side == OrderSide.BUY else "sell",
                "Quantity": order.quantity,
                "BidPrice": order.price if order.price else 0,
                "OfferPrice": order.price if order.price else 0,
                "AuditId": f"order_{int(time.time())}",
                "TradingAccountId": self.user_account.get("TradingAccountId") if self.user_account else 0,
                "IfDone": [],
                "Close": []
            }
            
            async with aiohttp.ClientSession() as session:
                headers = self._get_headers()
                
                if order.order_type == OrderType.MARKET:
                    endpoint = f"{self.api_base_url}/order/newtradeorder"
                else:
                    endpoint = f"{self.api_base_url}/order/newstoplimitorder"
                    
                self.logger.info(f"Calling place order endpoint: {endpoint}")
                
                async with session.post(
                    endpoint,
                    headers=headers,
                    json=order_request
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        order_id = str(data.get("OrderId", ""))
                        self.logger.info(f"Order placed successfully: {order_id}")
                        return order_id
                    else:
                        error_text = await response.text()
                        raise Exception(f"Order placement failed: {response.status} - {error_text}")
                        
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
        if not self._authenticated:
            await self.authenticate()
            
        try:
            async with aiohttp.ClientSession() as session:
                headers = self._get_headers()
                
                cancel_request = {
                    "OrderId": int(order_id),
                    "TradingAccountId": self.user_account.get("TradingAccountId") if self.user_account else 0
                }
                
                endpoint = f"{self.api_base_url}/order/cancel"
                self.logger.info(f"Calling cancel order endpoint: {endpoint}")
                
                async with session.post(
                    endpoint,
                    headers=headers,
                    json=cancel_request
                ) as response:
                    if response.status == 200:
                        self.logger.info(f"Order cancelled successfully: {order_id}")
                        return True
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Order cancellation failed: {response.status} - {error_text}")
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
        if not self._authenticated:
            await self.authenticate()
            
        try:
            # Get market ID for symbol
            market_id = self._get_market_id(symbol)
            
            # Map timeframes to GainCapital intervals
            interval_map = {
                "5m": ("MINUTE", 5),
                "15m": ("MINUTE", 15), 
                "1h": ("HOUR", 1),
                "4h": ("HOUR", 4),
                "1d": ("DAY", 1)
            }
            interval, span = interval_map.get(timeframe, ("HOUR", 1))
            
            async with aiohttp.ClientSession() as session:
                headers = self._get_headers()
                
                params = {
                    "interval": interval,
                    "span": span,
                    "PriceBars": bars,
                    "PriceType": "MID"
                }
                
                endpoint = f"{self.api_base_url}/market/{market_id}/barhistory"
                
                # Debug headers being sent
                self.logger.info(f"Headers being sent: {headers}")
                
                async with session.get(
                    endpoint,
                    headers=headers,
                    params=params
                ) as response:
                    self.logger.info(f"Historical data response status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        bars = data.get("PriceBars", [])
                        
                        rows = []
                        for bar in bars:
                            # Parse Microsoft .NET JSON date format: /Date(timestamp)/
                            date_str = bar["BarDate"]
                            if "/Date(" in date_str:
                                timestamp_ms = int(date_str.split("(")[1].split(")")[0])
                                timestamp = pd.to_datetime(timestamp_ms, unit='ms')
                            else:
                                timestamp = pd.to_datetime(date_str)
                            
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
                        if not df.empty:
                            df.set_index("timestamp", inplace=True)
                        return df
                    else:
                        raise Exception(f"API request failed: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            raise
    
    async def get_live_price(self, symbol: str) -> Dict:
        """
        Get current live price using GainCapital API v2.
        
        Args:
            symbol: Symbol in common format
            
        Returns:
            Dictionary with bid, ask, spread info
        """
        if not self._authenticated:
            await self.authenticate()
            
        try:
            gc_symbol = self.symbol_mapper.to_broker_symbol(symbol)
            
            async with aiohttp.ClientSession() as session:
                headers = self._get_headers()
                
                endpoint = f"{self.api_base_url}/market/{gc_symbol}/information"
                self.logger.info(f"Calling live price endpoint: {endpoint}")
                
                async with session.get(
                    endpoint,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        market_info = data.get("MarketInformation", {})
                        
                        bid = float(market_info.get("CurrentBid", {}).get("Price", 0))
                        ask = float(market_info.get("CurrentAsk", {}).get("Price", 0))
                        
                        return {
                            "symbol": symbol,
                            "bid": bid,
                            "ask": ask,
                            "mid": (bid + ask) / 2 if bid and ask else 0,
                            "spread": ask - bid if bid and ask else 0,
                            "timestamp": datetime.now()
                        }
                    else:
                        raise Exception(f"API request failed: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Error getting live price: {e}")
            raise
    
    def map_symbol_to_broker(self, common_symbol: str) -> str:
        """Map common symbol to GainCapital format."""
        return self.symbol_mapper.to_broker_symbol(common_symbol)
    
    def map_symbol_from_broker(self, broker_symbol: str) -> str:
        """Map GainCapital symbol to common format."""
        return self.symbol_mapper.from_broker_symbol(broker_symbol) 