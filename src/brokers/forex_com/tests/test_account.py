"""Tests for the Forex.com AccountHandler.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from src.brokers.forex_com.account import AccountHandler
from src.brokers.forex_com.api import ApiClient
from src.brokers.forex_com.types import ForexComApiResponseKeys


@pytest.fixture
def mock_api_client():
    """Fixture to create a mock ApiClient."""
    return Mock(spec=ApiClient)

@pytest.fixture
def account_handler(mock_api_client):
    """Fixture to create an AccountHandler with a mocked ApiClient."""
    return AccountHandler(api_client=mock_api_client)

@pytest.fixture
def mock_account_response():
    """Mock a successful account info API response."""
    return {
        ForexComApiResponseKeys.TRADING_ACCOUNTS: [
            {
                "TradingAccountId": 12345,
                "AccountBalance": 50000.0,
                "AccountCurrency": "USD",
                "AvailableFunds": 45000.0,
                "MarginRequirement": 5000.0,
                "UnrealizedPnL": 150.0,
                "AccountName": "Test Account",
                "AccountStatus": "Active"
            }
        ]
    }

@pytest.mark.asyncio
async def test_get_account_info_success(account_handler, mock_api_client, mock_account_response):
    """Test successfully retrieving account information."""
    # Configure the mock ApiClient to return a successful response
    mock_api_client._make_request = AsyncMock(return_value=(200, mock_account_response))

    account_info = await account_handler.get_account_info()

    # Verify that the correct API endpoint was called
    mock_api_client._make_request.assert_called_once_with('GET', '/useraccount/ClientAndTradingAccount')

    # Verify the returned data is correctly parsed
    assert account_info["account_id"] == 12345
    assert account_info["balance"] == 50000.0
    assert account_info["currency"] == "USD"
    assert account_info["available_funds"] == 45000.0
    assert account_info["unrealized_pnl"] == 150.0

@pytest.mark.asyncio
async def test_get_account_info_no_accounts(account_handler, mock_api_client):
    """Test the case where the API returns no trading accounts."""
    # Configure the mock to return an empty list of accounts
    empty_response = {ForexComApiResponseKeys.TRADING_ACCOUNTS: []}
    mock_api_client._make_request = AsyncMock(return_value=(200, empty_response))

    account_info = await account_handler.get_account_info()

    assert account_info == {}

@pytest.mark.asyncio
async def test_get_account_info_api_failure(account_handler, mock_api_client):
    """Test handling of an API failure when getting account info."""
    # Configure the mock to simulate an API error
    mock_api_client._make_request = AsyncMock(return_value=(500, "Internal Server Error"))

    with pytest.raises(Exception, match="API request failed: 500 - Internal Server Error"):
        await account_handler.get_account_info()
