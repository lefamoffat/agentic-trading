"""
Tests for the Forex.com OrderHandler.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from src.brokers.forex_com.orders import OrderHandler
from src.brokers.forex_com.api import ApiClient
from src.brokers.symbol_mapper import SymbolMapper, BrokerType
from src.brokers.base import Order, OrderType, OrderSide, OrderStatus

@pytest.fixture
def mock_api_client():
    mock = Mock(spec=ApiClient)
    # Mock the nested auth_handler and its user_account attribute
    mock.auth_handler = Mock()
    mock.auth_handler.user_account = {"TradingAccountId": 12345}
    return mock

@pytest.fixture
def symbol_mapper():
    return SymbolMapper(BrokerType.FOREX_COM)

@pytest.fixture
def order_handler(mock_api_client, symbol_mapper):
    return OrderHandler(api_client=mock_api_client, symbol_mapper=symbol_mapper)

@pytest.fixture
def mock_orders_response():
    return {
        "ActiveOrders": [
            {"OrderId": "1", "Market": {"Name": "EUR_USD"}, "Quantity": 5000.0, "Type": "Limit", "TriggerPrice": 1.1},
            {"OrderId": "2", "Market": {"Name": "USD_JPY"}, "Quantity": -8000.0, "Type": "Stop", "TriggerPrice": 150.0},
        ]
    }

@pytest.mark.asyncio
async def test_get_orders_success(order_handler, mock_api_client, mock_orders_response):
    mock_api_client._make_request.return_value = (200, mock_orders_response)
    
    orders = await order_handler.get_orders()
    
    mock_api_client._make_request.assert_called_once_with('GET', '/order/activeorders')
    assert len(orders) == 2
    
    eur_order = next(o for o in orders if o.symbol == "EUR/USD")
    assert eur_order.side == OrderSide.BUY
    assert eur_order.order_type == OrderType.LIMIT
    
    jpy_order = next(o for o in orders if o.symbol == "USD/JPY")
    assert jpy_order.side == OrderSide.SELL
    assert jpy_order.order_type == OrderType.STOP

@pytest.mark.asyncio
async def test_place_order_success(order_handler, mock_api_client):
    order_to_place = Order(symbol="EUR/USD", quantity=100, order_type=OrderType.MARKET, side=OrderSide.BUY)
    mock_api_client.get_market_id = AsyncMock(return_value="12345")
    mock_api_client._make_request.return_value = (200, {"OrderId": "555"})
    
    order_id = await order_handler.place_order(order_to_place)
    
    mock_api_client.get_market_id.assert_called_once_with("EUR/USD")
    mock_api_client._make_request.assert_called_once()
    assert order_id == "555"

@pytest.mark.asyncio
async def test_place_order_not_market(order_handler):
    order_to_place = Order(symbol="EUR/USD", quantity=100, order_type=OrderType.LIMIT, side=OrderSide.BUY)
    
    with pytest.raises(NotImplementedError):
        await order_handler.place_order(order_to_place)

@pytest.mark.asyncio
async def test_cancel_order_success(order_handler, mock_api_client):
    mock_api_client._make_request.return_value = (200, {"Success": True}) # Simplified response
    
    result = await order_handler.cancel_order("123")
    
    assert result is True
    mock_api_client._make_request.assert_called_once()
    call_args = mock_api_client._make_request.call_args[1]['json_data']
    assert call_args['OrderId'] == 123

@pytest.mark.asyncio
async def test_cancel_order_failure(order_handler, mock_api_client):
    mock_api_client._make_request.return_value = (400, {"Error": "Bad Request"})
    
    result = await order_handler.cancel_order("123")
    
    assert result is False
