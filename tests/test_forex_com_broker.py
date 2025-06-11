"""
Tests for Forex.com broker integration using GainCapital API.
"""

import pytest
import asyncio
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

import pandas as pd

from src.brokers.forex_com import ForexComBroker
from src.brokers.base import Order, OrderType, OrderSide, OrderStatus


class TestForexComBroker:
    """Test cases for Forex.com broker using GainCapital API."""
    
    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker instance."""
        return ForexComBroker(
            api_key="test_username",
            api_secret="test_password", 
            sandbox=True
        )
    
    @pytest.fixture
    def mock_session_response(self):
        """Mock successful GainCapital login response."""
        return {
            "Session": "test_session_token_123456",
            "SessionId": "session_id_789",
            "UserAccount": {
                "TradingAccountId": 12345,
                "UserId": 67890
            }
        }
    
    @pytest.fixture
    def mock_account_response(self):
        """Mock GainCapital account info response."""
        return {
            "TradingAccounts": [
                {
                    "TradingAccountId": 12345,
                    "AccountBalance": 50000.0,
                    "AccountCurrency": "USD",
                    "AvailableFunds": 45000.0,
                    "MarginRequirement": 5000.0,
                    "TotalMarginRequirement": 5000.0,
                    "AccountName": "Test Account",
                    "AccountStatus": "Active"
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_authentication_success(self, mock_broker, mock_session_response):
        """Test successful authentication with GainCapital API."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json.return_value = asyncio.Future()
            mock_response.json.return_value.set_result(mock_session_response)
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await mock_broker.authenticate()
            
            assert result is True
            assert mock_broker._authenticated is True
            assert mock_broker.session_token == "test_session_token_123456"
            assert mock_broker.session_id == "session_id_789"
            assert mock_broker.user_account["TradingAccountId"] == 12345
    
    @pytest.mark.asyncio
    async def test_authentication_failure(self, mock_broker):
        """Test authentication failure."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 401
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await mock_broker.authenticate()
            
            assert result is False
            assert mock_broker._authenticated is False
            assert mock_broker.session_token is None
    
    @pytest.mark.asyncio
    async def test_get_account_info(self, mock_broker, mock_session_response, mock_account_response):
        """Test getting account information."""
        mock_broker._authenticated = True
        mock_broker.session_token = "test_token"
        mock_broker.user_account = mock_session_response["UserAccount"]
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json.return_value = asyncio.Future()
            mock_response.json.return_value.set_result(mock_account_response)
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            account_info = await mock_broker.get_account_info()
            
            assert account_info["account_id"] == 12345
            assert account_info["balance"] == 50000.0
            assert account_info["currency"] == "USD"
            assert account_info["available_funds"] == 45000.0
    
    @pytest.mark.asyncio 
    async def test_get_positions(self, mock_broker):
        """Test getting current positions."""
        mock_broker._authenticated = True
        mock_broker.session_token = "test_token"
        
        mock_positions_response = {
            "OpenPositions": [
                {
                    "Market": {"Name": "EUR_USD"},
                    "Quantity": 10000.0,
                    "Price": 1.1000,
                    "PnL": 150.0
                },
                {
                    "Market": {"Name": "GBP_USD"}, 
                    "Quantity": -5000.0,
                    "Price": 1.2500,
                    "PnL": -75.0
                }
            ]
        }
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json.return_value = asyncio.Future()
            mock_response.json.return_value.set_result(mock_positions_response)
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            positions = await mock_broker.get_positions()
            
            assert len(positions) == 2
            assert positions[0].symbol == "EUR/USD"
            assert positions[0].quantity == 10000.0
            assert positions[1].symbol == "GBP/USD"
            assert positions[1].quantity == -5000.0
    
    @pytest.mark.asyncio
    async def test_get_orders(self, mock_broker):
        """Test getting active orders."""
        mock_broker._authenticated = True
        mock_broker.session_token = "test_token"
        
        mock_orders_response = {
            "ActiveOrders": [
                {
                    "OrderId": 123456,
                    "Market": {"Name": "EUR_USD"},
                    "Quantity": 5000.0,
                    "Type": "Limit Order",
                    "TriggerPrice": 1.0950
                },
                {
                    "OrderId": 789012,
                    "Market": {"Name": "USD_JPY"},
                    "Quantity": -8000.0,
                    "Type": "Stop Order", 
                    "TriggerPrice": 149.50
                }
            ]
        }
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json.return_value = asyncio.Future()
            mock_response.json.return_value.set_result(mock_orders_response)
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            orders = await mock_broker.get_orders()
            
            assert len(orders) == 2
            assert orders[0].symbol == "EUR/USD"
            assert orders[0].order_type == OrderType.LIMIT
            assert orders[0].side == OrderSide.BUY
            assert orders[1].symbol == "USD/JPY"
            assert orders[1].order_type == OrderType.STOP
            assert orders[1].side == OrderSide.SELL
    
    @pytest.mark.asyncio
    async def test_place_market_order(self, mock_broker):
        """Test placing a market order."""
        mock_broker._authenticated = True
        mock_broker.session_token = "test_token"
        mock_broker.user_account = {"TradingAccountId": 12345}
        
        mock_order_response = {
            "OrderId": 555666,
            "Status": "Accepted"
        }
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json.return_value = asyncio.Future()
            mock_response.json.return_value.set_result(mock_order_response)
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            order = Order(
                symbol="EUR/USD",
                quantity=10000,
                order_type=OrderType.MARKET,
                side=OrderSide.BUY
            )
            
            order_id = await mock_broker.place_order(order)
            
            assert order_id == "555666"
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, mock_broker):
        """Test cancelling an order."""
        mock_broker._authenticated = True
        mock_broker.session_token = "test_token"
        mock_broker.user_account = {"TradingAccountId": 12345}
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await mock_broker.cancel_order("123456")
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_get_historical_data(self, mock_broker):
        """Test getting historical price data."""
        mock_broker._authenticated = True
        mock_broker.session_token = "test_token"
        
        mock_history_response = {
            "PriceBars": [
                {
                    "BarDate": "2024-01-01T10:00:00Z",
                    "Open": 1.1000,
                    "High": 1.1050,
                    "Low": 1.0990,
                    "Close": 1.1020,
                    "Volume": 1000
                },
                {
                    "BarDate": "2024-01-01T11:00:00Z", 
                    "Open": 1.1020,
                    "High": 1.1080,
                    "Low": 1.1010,
                    "Close": 1.1060,
                    "Volume": 1200
                }
            ]
        }
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json.return_value = asyncio.Future()
            mock_response.json.return_value.set_result(mock_history_response)
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 2)
            
            df = await mock_broker.get_historical_data("EUR/USD", "1h", start_date, end_date)
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert "open" in df.columns
            assert "close" in df.columns
            assert df.iloc[0]["open"] == 1.1000
            assert df.iloc[1]["close"] == 1.1060
    
    @pytest.mark.asyncio
    async def test_get_live_price(self, mock_broker):
        """Test getting live price data."""
        mock_broker._authenticated = True
        mock_broker.session_token = "test_token"
        
        mock_price_response = {
            "MarketInformation": {
                "CurrentBid": {"Price": 1.0995},
                "CurrentAsk": {"Price": 1.1005}
            }
        }
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json.return_value = asyncio.Future()
            mock_response.json.return_value.set_result(mock_price_response)
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            price_data = await mock_broker.get_live_price("EUR/USD")
            
            assert price_data["symbol"] == "EUR/USD"
            assert price_data["bid"] == 1.0995
            assert price_data["ask"] == 1.1005
            assert price_data["mid"] == 1.1000
            assert price_data["spread"] == 0.001
    
    def test_symbol_mapping(self, mock_broker):
        """Test symbol mapping between common and GainCapital formats."""
        assert mock_broker.map_symbol_to_broker("EUR/USD") == "EUR_USD"
        assert mock_broker.map_symbol_to_broker("GBP/USD") == "GBP_USD"
        
        assert mock_broker.map_symbol_from_broker("EUR_USD") == "EUR/USD"
        assert mock_broker.map_symbol_from_broker("GBP_USD") == "GBP/USD"


@pytest.mark.integration
class TestForexComBrokerIntegration:
    """Integration tests for Forex.com broker - requires real credentials."""
    
    @pytest.fixture
    def real_broker(self):
        """Create a real broker instance using environment variables."""
        api_key = os.getenv("FOREX_COM_USERNAME")
        api_secret = os.getenv("FOREX_COM_PASSWORD")
        
        if not api_key or not api_secret:
            pytest.skip("Missing FOREX_COM credentials in environment")
        
        return ForexComBroker(api_key, api_secret, sandbox=True)
    
    @pytest.mark.asyncio
    async def test_real_authentication(self, real_broker):
        """Test authentication with real GainCapital API."""
        result = await real_broker.authenticate()
        assert result is True
        assert real_broker._authenticated is True
        assert real_broker.session_token is not None
    
    @pytest.mark.asyncio
    async def test_real_account_info(self, real_broker):
        """Test getting real account information."""
        await real_broker.authenticate()
        account_info = await real_broker.get_account_info()
        
        assert "account_id" in account_info
        assert "balance" in account_info
        assert "currency" in account_info
        assert isinstance(account_info["balance"], float)
    
    @pytest.mark.asyncio
    async def test_real_live_price(self, real_broker):
        """Test getting real live price data."""
        await real_broker.authenticate()
        price_data = await real_broker.get_live_price("EUR/USD")
        
        assert price_data["symbol"] == "EUR/USD"
        assert price_data["bid"] > 0
        assert price_data["ask"] > 0
        assert price_data["ask"] > price_data["bid"] 