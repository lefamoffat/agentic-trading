"""Broker factory for routing to different broker implementations.

This factory provides a central point for creating broker instances,
making it easy to add new brokers without changing existing code.
"""

from typing import ClassVar, Dict, Type

from ..types import BrokerType
from ..utils.logger import get_logger
from .base import BaseBroker
from .forex_com import ForexComBroker


class BrokerFactory:
    """Factory for creating broker instances."""

    # Registry of available brokers
    _broker_registry: ClassVar[Dict[BrokerType, Type[BaseBroker]]] = {
        BrokerType.FOREX_COM: ForexComBroker,
    }

    def __init__(self):
        self.logger = get_logger(__name__)

    @classmethod
    def register_broker(cls, broker_type: BrokerType, broker_class: Type[BaseBroker]) -> None:
        """Register a new broker implementation.

        Args:
            broker_type: The broker type identifier
            broker_class: The broker implementation class

        """
        cls._broker_registry[broker_type] = broker_class

    @classmethod
    def get_available_brokers(cls) -> list[str]:
        """Get list of available broker names.

        Returns:
            List of broker names as strings

        """
        return [broker.value for broker in cls._broker_registry.keys()]

    def create_broker(
        self,
        broker_name: str,
        api_key: str,
        api_secret: str,
        sandbox: bool = True,
        **kwargs
    ) -> BaseBroker:
        """Create a broker instance based on broker name.

        Args:
            broker_name: Name of the broker (e.g., "forex.com")
            api_key: API key or username
            api_secret: API secret or password
            sandbox: Whether to use sandbox/demo environment
            **kwargs: Additional broker-specific parameters

        Returns:
            Configured broker instance

        Raises:
            ValueError: If broker is not supported

        """
        try:
            broker_type = BrokerType(broker_name)
        except ValueError as err:
            available = self.get_available_brokers()
            raise ValueError(
                f"Unsupported broker: '{broker_name}'. "
                f"Available brokers: {available}"
            ) from err

        broker_class = self._broker_registry[broker_type]

        self.logger.info(f"Creating {broker_name} broker instance")

        try:
            return broker_class(
                api_key=api_key,
                api_secret=api_secret,
                sandbox=sandbox,
                **kwargs
            )
        except Exception as e:
            self.logger.error(f"Failed to create {broker_name} broker: {e}")
            raise

    def is_broker_supported(self, broker_name: str) -> bool:
        """Check if a broker is supported.

        Args:
            broker_name: Name of the broker to check

        Returns:
            True if broker is supported, False otherwise

        """
        try:
            BrokerType(broker_name)
            return True
        except ValueError:
            return False


# Global factory instance
broker_factory = BrokerFactory()
