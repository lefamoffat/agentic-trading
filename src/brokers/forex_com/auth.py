"""
Authentication handler for Forex.com broker.
"""

import os
from typing import Dict, Any, Optional
import json

import aiohttp

from src.utils.logger import get_logger
from src.brokers.exceptions import JsonParseError


class AuthenticationHandler:
    """Handles authentication with GainCapital API v2."""

    AUTH_BASE_URL = "https://ciapi.cityindex.com/v2"

    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize the authentication handler.

        Args:
            api_key: GainCapital username.
            api_secret: GainCapital password.
        """
        self.username = api_key
        self.password = api_secret
        self.app_key = os.getenv('FOREX_COM_APP_KEY')
        if not self.app_key:
            raise ValueError("FOREX_COM_APP_KEY environment variable is required")

        self.logger = get_logger(__name__)
        self.session_token: Optional[str] = None
        self.user_account: Optional[Dict[str, Any]] = None
        self._authenticated: bool = False

    @property
    def is_authenticated(self) -> bool:
        """Check if the session is authenticated."""
        return self._authenticated and self.session_token is not None

    async def authenticate(self) -> bool:
        """
        Authenticate with GainCapital API v2.

        Returns:
            True if authentication is successful, False otherwise.
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

                auth_endpoint = f"{self.AUTH_BASE_URL}/Session"
                self.logger.info(f"Authenticating to endpoint: {auth_endpoint}")
                self.logger.debug(f"Login data: {dict((k, v if k != 'Password' else '***') for k, v in login_data.items())}")

                async with session.request("POST", auth_endpoint, json=login_data, headers=headers) as response:
                    try:
                        data = await response.json(content_type=None)
                    except (json.JSONDecodeError, aiohttp.ContentTypeError) as e:
                        raw_text = await response.text()
                        self.logger.error(
                            f"Failed to decode JSON from auth response. Status: {response.status}, Body: {raw_text[:200]}..."
                        )
                        raise JsonParseError(
                            f"Auth endpoint returned non-JSON response. Status: {response.status}"
                        ) from e
                    
                    self.logger.debug(f"Authentication response status: {response.status}")
                    self.logger.debug(f"Authentication response data: {data}")

                    if response.status == 200:
                        status_code = data.get("statusCode", -1)
                        if status_code == 0:
                            self.session_token = data.get("session")
                            if "UserAccount" in data:
                                self.user_account = data["UserAccount"]

                            if self.session_token:
                                self._authenticated = True
                                self.logger.info(f"Successfully authenticated with Forex.com. Session: {self.session_token[:8]}...")
                                return True
                        else:
                            self.logger.error(f"Authentication failed with API statusCode: {status_code}")
                            return False

                    self.logger.error(f"Authentication failed with HTTP status: {response.status}")
                    return False

        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return False

    def get_headers(self) -> Dict[str, str]:
        """
        Get headers with session token for authenticated requests.

        Returns:
            A dictionary of headers for authenticated requests.

        Raises:
            Exception: If not authenticated.
        """
        if not self.is_authenticated:
            raise Exception("Not authenticated - call authenticate() first")

        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Session": self.session_token,
            "UserName": self.username
        } 