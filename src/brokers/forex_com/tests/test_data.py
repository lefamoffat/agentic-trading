"""Tests for the Forex.com DataHandler.
"""

from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from src.brokers.forex_com.api import ApiClient
from src.brokers.forex_com.data import DataHandler
from src.brokers.forex_com.types import ForexComApiResponseKeys
from src.types import Timeframe

@pytest.fixture
def mock_api_client():
    """Fixture to create a mock ApiClient."""
    mock = Mock(spec=ApiClient)
    mock.get_market_id = AsyncMock(return_value="12345")
    return mock

@pytest.fixture
def data_handler(mock_api_client):
    """Fixture to create a DataHandler with a mocked ApiClient."""
    return DataHandler(api=mock_api_client)

@pytest.fixture
def mock_history_response():
    """Mock a successful historical data API response."""
    return {
        ForexComApiResponseKeys.PRICE_BARS: [
            {"BarDate": "/Date(1704067200000)/", "Open": 1.1, "High": 1.2, "Low": 1.0, "Close": 1.15, "Volume": 100},
            {"BarDate": "/Date(1704153600000)/", "Open": 1.15, "High": 1.25, "Low": 1.12, "Close": 1.22, "Volume": 120},
        ]
    }

@pytest.mark.asyncio
async def test_get_historical_data_success(data_handler, mock_api_client, mock_history_response):
    """Test successfully retrieving and parsing historical data."""
    mock_api_client._make_request.return_value = (200, mock_history_response)

    df = await data_handler.get_historical_data(
        symbol="EUR/USD", timeframe=Timeframe.H1.value, bars=2
    )

    mock_api_client._make_request.assert_called_once()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "open" in df.columns
    assert df.iloc[0]["open"] == 1.1
    assert df.iloc[1]["close"] == 1.22

@pytest.mark.asyncio
async def test_get_live_price_success(data_handler, mock_api_client):
    """Test successfully retrieving a live price."""
    hist_response = {ForexComApiResponseKeys.PRICE_BARS: [{"BarDate": "/Date(1704067200000)/", "Close": 1.25}]}
    info_response = {"MarketInformation": {"MarketSpreads": [{"Spread": 0.0002}]}}

    # Mock the two separate API calls made by get_live_price
    mock_api_client._make_request = AsyncMock(side_effect=[
        (200, hist_response),
        (200, info_response)
    ])

    price_data = await data_handler.get_live_price("EUR/USD")

    mock_api_client.get_market_id.assert_called_once_with("EUR/USD")
    assert mock_api_client._make_request.call_count == 2

    assert price_data["symbol"] == "EUR/USD"
    assert price_data["mid"] == 1.25
    assert price_data["spread"] == 0.0002
    assert pytest.approx(price_data["bid"]) == 1.2499 # 1.25 - 0.0002/2
    assert pytest.approx(price_data["ask"]) == 1.2501 # 1.25 + 0.0002/2

def test_parse_dotnet_date(data_handler):
    """Test the parsing of the proprietary .NET date format."""
    date_str = "/Date(1609459200000)/" # This is 2021-01-01 00:00:00 UTC
    parsed_date = data_handler._parse_dotnet_date(date_str)
    assert parsed_date.year == 2021
    assert parsed_date.month == 1
    assert parsed_date.day == 1
