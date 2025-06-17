"""Account handler for Forex.com broker.
"""

from typing import Dict

from src.brokers.forex_com.api import ApiClient
from src.brokers.forex_com.types import ForexComApiResponseKeys
from src.utils.logger import get_logger


class AccountHandler:
    """Handles account-related operations."""

    def __init__(self, api_client: ApiClient):
        """Initialize the account handler.

        Args:
            api_client: The API client instance.

        """
        self.api_client = api_client
        self.logger = get_logger(__name__)

    async def get_account_info(self) -> Dict:
        """Get account information using GainCapital API v2.

        Returns:
            A dictionary with account info.

        Raises:
            Exception: If the API request fails.

        """
        try:
            endpoint = "/useraccount/ClientAndTradingAccount"
            status, data = await self.api_client._make_request('GET', endpoint)

            if status == 200:
                trading_accounts = data.get(ForexComApiResponseKeys.TRADING_ACCOUNTS, [])
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
                    self.logger.warning("No trading accounts found in API response.")
                    return {}
            else:
                raise Exception(f"API request failed: {status} - {data}")

        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            raise
