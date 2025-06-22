"""Integration tests for the Forex.com broker.

These tests require real credentials to be set in the environment:
- FOREX_COM_USERNAME
- FOREX_COM_PASSWORD
- FOREX_COM_APP_KEY

To run, use: pytest -m integration
"""

import os

import pytest
import pytest_asyncio

from src.brokers.forex_com.broker import ForexComBroker
from src.types import Timeframe

@pytest.mark.integration
class TestForexComBrokerIntegration:
    """Integration tests for the Forex.com broker, requiring real credentials."""

    @pytest.fixture(scope="class")
    def real_broker(self):
        """Creates a real broker instance using credentials from environment variables.
        Skips the tests if credentials are not available.
        """
        api_key = os.getenv("FOREX_COM_USERNAME")
        api_secret = os.getenv("FOREX_COM_PASSWORD")
        app_key = os.getenv("FOREX_COM_APP_KEY")

        if not all([api_key, api_secret, app_key]):
            pytest.skip("Missing FOREX_COM credentials in environment variables.")

        return ForexComBroker(api_key=api_key, api_secret=api_secret, sandbox=True)

    @pytest_asyncio.fixture(scope="class", autouse=True)
    async def authenticated_broker(self, real_broker):
        """Fixture to ensure the broker is authenticated before running tests in this class.
        This runs once for the class.
        """
        await real_broker.authenticate()
        yield real_broker

    @pytest.mark.asyncio
    async def test_real_authentication_is_successful(self, authenticated_broker):
        """Verify that the broker is authenticated."""
        assert authenticated_broker.auth_handler.is_authenticated is True
        assert authenticated_broker.auth_handler.session_token is not None

    @pytest.mark.asyncio
    async def test_real_get_account_info(self, authenticated_broker):
        """Test getting real account information."""
        account_info = await authenticated_broker.get_account_info()

        assert "account_id" in account_info
        assert "balance" in account_info
        assert isinstance(account_info["balance"], float)
        assert account_info["balance"] >= 0

    @pytest.mark.asyncio
    async def test_real_get_live_price(self, authenticated_broker):
        """Test getting real live price data for a common symbol."""
        price_data = await authenticated_broker.get_live_price("EUR/USD")

        assert price_data["symbol"] == "EUR/USD"
        assert price_data["bid"] > 0
        assert price_data["ask"] > 0
        assert price_data["ask"] > price_data["bid"]

    @pytest.mark.asyncio
    async def test_real_get_historical_data(self, authenticated_broker):
        """Test getting real historical data."""
        df = await authenticated_broker.get_historical_data(
            "EUR/USD", Timeframe.H1.value, bars=10
        )

        assert not df.empty
        assert len(df) == 10
        assert "open" in df.columns
        assert "close" in df.columns
