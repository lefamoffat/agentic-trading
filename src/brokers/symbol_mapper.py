"""Symbol mapping utilities for broker integrations.
This provides a middle layer to map between common symbol formats
and broker-specific naming conventions.
"""


from ..types import BrokerType


class SymbolMapper:
    """Maps between common symbol formats and broker-specific formats.
    
    Common format: "EUR/USD", "GBP/USD", etc.
    """

    # Forex.com uses specific naming conventions
    FOREX_COM_MAPPINGS = {
        # Common -> Forex.com format
        "EUR/USD": "EUR_USD",
        "GBP/USD": "GBP_USD",
        "USD/JPY": "USD_JPY",
        "USD/CHF": "USD_CHF",
        "USD/CAD": "USD_CAD",
        "AUD/USD": "AUD_USD",
        "NZD/USD": "NZD_USD",
        "EUR/GBP": "EUR_GBP",
        "EUR/JPY": "EUR_JPY",
        "EUR/CHF": "EUR_CHF",
        "EUR/CAD": "EUR_CAD",
        "EUR/AUD": "EUR_AUD",
        "GBP/JPY": "GBP_JPY",
        "GBP/CHF": "GBP_CHF",
        "GBP/CAD": "GBP_CAD",
        "GBP/AUD": "GBP_AUD",
        "CHF/JPY": "CHF_JPY",
        "CAD/JPY": "CAD_JPY",
        "AUD/JPY": "AUD_JPY",
        "NZD/JPY": "NZD_JPY",
        "AUD/CAD": "AUD_CAD",
        "AUD/CHF": "AUD_CHF",
        "CAD/CHF": "CAD_CHF",
        "NZD/CAD": "NZD_CAD",
        "NZD/CHF": "NZD_CHF",
    }

    # Reverse mapping for Forex.com
    FOREX_COM_REVERSE_MAPPINGS = {v: k for k, v in FOREX_COM_MAPPINGS.items()}

    def __init__(self, broker_type: BrokerType = BrokerType.FOREX_COM):
        """Initialize symbol mapper for specific broker.
        
        Args:
            broker_type: The broker type to map symbols for

        """
        self.broker_type = broker_type

        if broker_type == BrokerType.FOREX_COM:
            self.to_broker_mappings = self.FOREX_COM_MAPPINGS
            self.from_broker_mappings = self.FOREX_COM_REVERSE_MAPPINGS
        else:
            # Generic/no mapping
            self.to_broker_mappings = {}
            self.from_broker_mappings = {}

    def to_broker_symbol(self, common_symbol: str) -> str:
        """Map common symbol format to broker-specific format.
        
        Args:
            common_symbol: Symbol in common format (e.g., "EUR/USD")
            
        Returns:
            Broker-specific symbol format
            
        Raises:
            ValueError: If symbol is not supported

        """
        if not self.to_broker_mappings:
            return common_symbol

        broker_symbol = self.to_broker_mappings.get(common_symbol)
        if broker_symbol is None:
            raise ValueError(
                f"Symbol '{common_symbol}' not supported for {self.broker_type.value}. "
                f"Supported symbols: {list(self.to_broker_mappings.keys())}"
            )
        return broker_symbol

    def from_broker_symbol(self, broker_symbol: str) -> str:
        """Map broker-specific symbol format to common format.
        
        Args:
            broker_symbol: Symbol in broker-specific format
            
        Returns:
            Common symbol format
            
        Raises:
            ValueError: If symbol is not supported

        """
        if not self.from_broker_mappings:
            return broker_symbol

        common_symbol = self.from_broker_mappings.get(broker_symbol)
        if common_symbol is None:
            raise ValueError(
                f"Broker symbol '{broker_symbol}' not recognized for {self.broker_type.value}. "
                f"Known broker symbols: {list(self.from_broker_mappings.keys())}"
            )
        return common_symbol

    def is_supported(self, common_symbol: str) -> bool:
        """Check if a symbol is supported for this broker.
        
        Args:
            common_symbol: Symbol in common format
            
        Returns:
            True if supported, False otherwise

        """
        if not self.to_broker_mappings:
            return True  # Generic mapper supports all
        return common_symbol in self.to_broker_mappings

    def get_supported_symbols(self) -> list[str]:
        """Get list of all supported symbols in common format.
        
        Returns:
            List of supported symbols

        """
        if not self.to_broker_mappings:
            return []  # Generic mapper - unlimited
        return list(self.to_broker_mappings.keys())

    @classmethod
    def add_custom_mapping(
        cls,
        broker_type: BrokerType,
        common_symbol: str,
        broker_symbol: str
    ) -> None:
        """Add a custom symbol mapping for a broker.
        
        Args:
            broker_type: The broker to add mapping for
            common_symbol: Symbol in common format
            broker_symbol: Symbol in broker-specific format

        """
        if broker_type == BrokerType.FOREX_COM:
            cls.FOREX_COM_MAPPINGS[common_symbol] = broker_symbol
            cls.FOREX_COM_REVERSE_MAPPINGS[broker_symbol] = common_symbol
