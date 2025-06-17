"""Tests for the Forex.com AuthenticationHandler.
"""

from unittest.mock import patch

import pytest

from src.brokers.forex_com.auth import AuthenticationHandler
from src.brokers.forex_com.tests.conftest import create_async_session_mock


@pytest.fixture
def auth_handler():
    """Fixture to create an AuthenticationHandler instance."""
    return AuthenticationHandler(api_key="test_user", api_secret="test_pass")

@pytest.fixture
def mock_session_response():
    """Mock a successful GainCapital login response."""
    return {
        "statusCode": 0,
        "session": "test_session_token_123456",
        "UserAccount": {"TradingAccountId": 12345}
    }

@pytest.mark.asyncio
@patch('aiohttp.ClientSession')
async def test_authentication_success(mock_session_class, auth_handler, mock_session_response):
    """Test a successful authentication flow."""
    mock_session_cm = create_async_session_mock(mock_session_response)
    mock_session_class.return_value = mock_session_cm

    result = await auth_handler.authenticate()

    assert result is True
    assert auth_handler.is_authenticated is True
    assert auth_handler.session_token == "test_session_token_123456"
    assert auth_handler.user_account["TradingAccountId"] == 12345

@pytest.mark.asyncio
@patch('aiohttp.ClientSession')
async def test_authentication_failure_status_code(mock_session_class, auth_handler):
    """Test an authentication failure due to a non-200 status code."""
    mock_session_cm = create_async_session_mock(None, status_code=401)
    mock_session_class.return_value = mock_session_cm

    result = await auth_handler.authenticate()

    assert result is False
    assert auth_handler.is_authenticated is False
    assert auth_handler.session_token is None

@pytest.mark.asyncio
@patch('aiohttp.ClientSession')
async def test_authentication_failure_api_error(mock_session_class, auth_handler):
    """Test an authentication failure due to an API error response."""
    error_response = {"statusCode": 1, "errorMessage": "Invalid credentials"}
    mock_session_cm = create_async_session_mock(error_response, status_code=200)
    mock_session_class.return_value = mock_session_cm

    result = await auth_handler.authenticate()

    assert result is False
    assert auth_handler.is_authenticated is False

def test_get_headers_unauthenticated(auth_handler):
    """Test that getting headers fails if not authenticated."""
    with pytest.raises(Exception, match="Not authenticated"):
        auth_handler.get_headers()

def test_get_headers_authenticated(auth_handler):
    """Test getting headers when authenticated."""
    auth_handler._authenticated = True
    auth_handler.session_token = "test_token"
    auth_handler.username = "test_user"

    headers = auth_handler.get_headers()

    assert headers["Session"] == "test_token"
    assert headers["UserName"] == "test_user"
    assert headers["Content-Type"] == "application/json"
