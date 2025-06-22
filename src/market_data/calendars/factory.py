"""Calendar factory for creating market calendar instances.

This factory provides dynamic calendar selection based on asset class,
making it easy to switch between forex, stocks, crypto without changing core logic.
"""

from typing import ClassVar, Dict, Type

from src.types import AssetClass
from src.utils.logger import get_logger
from src.market_data.calendars.base import BaseCalendar
from src.market_data.calendars.forex import ForexCalendar

class CalendarFactory:
    """Factory for creating calendar instances based on asset class."""

    # Registry of available calendars
    _calendar_registry: ClassVar[Dict[AssetClass, Type[BaseCalendar]]] = {
        AssetClass.FOREX: ForexCalendar,
    }

    def __init__(self):
        self.logger = get_logger(__name__)

    @classmethod
    def register_calendar(cls, asset_class: AssetClass, calendar_class: Type[BaseCalendar]) -> None:
        """Register a new calendar implementation.

        Args:
            asset_class: The asset class identifier
            calendar_class: The calendar implementation class

        """
        cls._calendar_registry[asset_class] = calendar_class

    @classmethod
    def get_available_asset_classes(cls) -> list[str]:
        """Get list of available asset class names.

        Returns:
            List of asset class names as strings

        """
        return [asset_class.value for asset_class in cls._calendar_registry.keys()]

    def create_calendar(
        self,
        asset_class: str,
        timezone: str = "UTC",
        **kwargs
    ) -> BaseCalendar:
        """Create a calendar instance based on asset class.

        Args:
            asset_class: Name of the asset class (e.g., "forex")
            timezone: Timezone for calendar operations
            **kwargs: Additional calendar-specific parameters

        Returns:
            Configured calendar instance

        Raises:
            ValueError: If asset class is not supported

        """
        try:
            asset_class_enum = AssetClass(asset_class.lower())
        except ValueError as err:
            available = self.get_available_asset_classes()
            raise ValueError(
                f"Unsupported asset class: '{asset_class}'. "
                f"Available asset classes: {available}"
            ) from err

        calendar_class = self._calendar_registry[asset_class_enum]

        self.logger.info(f"Creating {asset_class} calendar instance")

        try:
            return calendar_class(
                timezone=timezone,
                **kwargs
            )
        except Exception as e:
            self.logger.error(f"Failed to create {asset_class} calendar: {e}")
            raise

    def is_asset_class_supported(self, asset_class: str) -> bool:
        """Check if an asset class is supported.

        Args:
            asset_class: Name of the asset class to check

        Returns:
            True if asset class is supported, False otherwise

        """
        try:
            AssetClass(asset_class.lower())
            return True
        except ValueError:
            return False

    def get_calendar_for_symbol(self, symbol: str, timezone: str = "UTC") -> BaseCalendar:
        """Get appropriate calendar for a given symbol.

        This method uses symbol patterns to determine the asset class
        and return the appropriate calendar.

        Args:
            symbol: Trading symbol (e.g., "EUR/USD", "AAPL", "BTC/USD")
            timezone: Timezone for calendar operations

        Returns:
            Appropriate calendar instance

        """
        # Determine asset class from symbol pattern
        if "/" in symbol and len(symbol.split("/")) == 2:
            # Forex pattern: EUR/USD, GBP/USD, etc.
            parts = symbol.split("/")
            if len(parts[0]) == 3 and len(parts[1]) == 3:
                asset_class = "forex"
            else:
                # Could be crypto: BTC/USD, ETH/USD
                asset_class = "crypto"  # Will fail for now, but ready for future
        else:
            # Likely stocks: AAPL, GOOGL, etc.
            asset_class = "stocks"  # Will fail for now, but ready for future

        self.logger.info(f"Detected asset class '{asset_class}' for symbol '{symbol}'")

        return self.create_calendar(asset_class, timezone)

# Global factory instance
calendar_factory = CalendarFactory() 