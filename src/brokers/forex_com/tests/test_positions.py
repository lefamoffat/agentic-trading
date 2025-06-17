"""Tests for the Forex.com PositionHandler.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from src.brokers.forex_com.api import ApiClient
from src.brokers.forex_com.positions import PositionHandler
from src.brokers.forex_com.types import ForexComApiResponseKeys
from src.brokers.symbol_mapper import BrokerType, SymbolMapper


@pytest.fixture
def mock_api_client():
    """Fixture to create a mock ApiClient."""
    return Mock(spec=ApiClient)

@pytest.fixture
def symbol_mapper():
    """Fixture to create a real SymbolMapper for Forex.com."""
    # It's often easier to use the real symbol mapper as it's pure logic
    return SymbolMapper(BrokerType.FOREX_COM)

@pytest.fixture
def position_handler(mock_api_client, symbol_mapper):
    """Fixture to create a PositionHandler with mocked dependencies."""
    return PositionHandler(api_client=mock_api_client, symbol_mapper=symbol_mapper)

@pytest.fixture
def mock_positions_response():
    """Mock a successful open positions API response."""
    return {
        ForexComApiResponseKeys.OPEN_POSITIONS: [
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
            },
            {
                "Market": {"Name": "UNKNOWN_SYMBOL"}, # A symbol our mapper doesn't know
                "Quantity": 100.0,
                "Price": 1.0,
                "PnL": 0.0
            }
        ]
    }

@pytest.mark.asyncio
async def test_get_positions_success(position_handler, mock_api_client, mock_positions_response):
    """Test successfully retrieving and parsing positions."""
    mock_api_client._make_request = AsyncMock(return_value=(200, mock_positions_response))

    positions = await position_handler.get_positions()

    # Verify the API call
    mock_api_client._make_request.assert_called_once_with('GET', '/order/openpositions')

    # Verify the results (should skip the unknown symbol)
    assert len(positions) == 2

    eur_pos = next(p for p in positions if p.symbol == "EUR/USD")
    gbp_pos = next(p for p in positions if p.symbol == "GBP/USD")

    assert eur_pos.quantity == 10000.0
    assert eur_pos.avg_price == 1.1000
    assert gbp_pos.quantity == -5000.0
    assert gbp_pos.avg_price == 1.2500

@pytest.mark.asyncio
async def test_get_positions_empty(position_handler, mock_api_client):
    """Test getting positions when there are none."""
    empty_response = {"OpenPositions": []}
    mock_api_client._make_request = AsyncMock(return_value=(200, empty_response))

    positions = await position_handler.get_positions()

    assert positions == []
