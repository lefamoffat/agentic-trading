"""
Tests for the Forex.com ApiClient.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, Mock, patch

from src.brokers.forex_com.api import ApiClient
from src.brokers.forex_com.auth import AuthenticationHandler
from src.brokers.forex_com.tests.conftest import create_async_session_mock

@pytest.fixture
def mock_auth_handler():
    """Fixture to create a mock AuthenticationHandler."""
    mock = MagicMock(spec=AuthenticationHandler)
    mock.is_authenticated = True
    mock.get_headers.return_value = {"Session": "test_token", "UserName": "test_user"}
    return mock

@pytest.fixture
def api_client(mock_auth_handler):
    """Fixture to create an ApiClient instance with a mocked auth handler."""
    return ApiClient(auth_handler=mock_auth_handler)

@pytest.mark.asyncio
@patch('aiohttp.ClientSession')
async def test_make_request_get_success(mock_session_class, api_client):
    """Test a successful GET request."""
    mock_response_data = {"status": "ok"}
    mock_session_cm = create_async_session_mock(mock_response_data)
    mock_session_class.return_value = mock_session_cm

    status, data = await api_client._make_request('GET', '/test/endpoint')

    assert status == 200
    assert data == mock_response_data
    api_client.auth_handler.get_headers.assert_called_once()

@pytest.mark.asyncio
@patch('aiohttp.ClientSession')
async def test_make_request_post_success(mock_session_class, api_client):
    """Test a successful POST request."""
    mock_response_data = {"status": "created"}
    mock_session_cm = create_async_session_mock(mock_response_data)
    mock_session_class.return_value = mock_session_cm

    status, data = await api_client._make_request('POST', '/test/endpoint', json_data={"key": "value"})

    assert status == 200
    assert data == mock_response_data

@pytest.mark.asyncio
async def test_get_market_id_success(api_client):
    """Test successfully retrieving a market ID."""
    mock_response = {"Markets": [{"MarketId": "12345"}]}
    
    with patch.object(api_client, '_make_request', new_callable=AsyncMock) as mock_make_request:
        mock_make_request.return_value = (200, mock_response)
        
        market_id = await api_client.get_market_id("EUR/USD")
        
        assert market_id == "12345"
        # Check that it's cached
        assert "EUR/USD" in api_client._market_id_cache
        assert api_client._market_id_cache["EUR/USD"] == "12345"

@pytest.mark.asyncio
async def test_get_market_id_cached(api_client):
    """Test that a cached market ID is returned without a new API call."""
    api_client._market_id_cache["EUR/USD"] = "54321"
    
    with patch.object(api_client, '_make_request', new_callable=AsyncMock) as mock_make_request:
        market_id = await api_client.get_market_id("EUR/USD")
        
        assert market_id == "54321"
        mock_make_request.assert_not_called()

@pytest.mark.asyncio
async def test_get_market_id_not_found(api_client):
    """Test the case where a market ID is not found for a symbol."""
    mock_response = {"Markets": []} # Empty response
    
    with patch.object(api_client, '_make_request', new_callable=AsyncMock) as mock_make_request:
        mock_make_request.return_value = (200, mock_response)
        
        with pytest.raises(ValueError, match="No market found for symbol: EUR/USD"):
            await api_client.get_market_id("EUR/USD") 