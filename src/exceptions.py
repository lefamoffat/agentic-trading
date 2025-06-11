"""
Project-wide custom exceptions for the agentic trading system.

This module defines specific exception types for different components
and error scenarios in the trading system.
"""

from typing import Optional, Any, Dict


class TradingSystemError(Exception):
    """Base exception class for all trading system errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """
        Initialize base trading system error.
        
        Args:
            message: Error message
            context: Additional context information
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
    
    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


# Data-related exceptions
class DataError(TradingSystemError):
    """Base class for data-related errors."""
    pass


class ValidationError(DataError):
    """Raised when data validation fails."""
    pass


class DataQualityError(DataError):
    """Raised when data quality is insufficient."""
    pass


class DataFormatError(DataError):
    """Raised when data format is incorrect."""
    pass


class MissingDataError(DataError):
    """Raised when required data is missing."""
    pass


class DataCorruptionError(DataError):
    """Raised when data appears to be corrupted."""
    pass


# Broker-related exceptions
class BrokerError(TradingSystemError):
    """Base class for broker-related errors."""
    pass


class BrokerConnectionError(BrokerError):
    """Raised when broker connection fails."""
    pass


class BrokerAuthenticationError(BrokerError):
    """Raised when broker authentication fails."""
    pass


class BrokerAPIError(BrokerError):
    """Raised when broker API returns an error."""
    
    def __init__(self, message: str, api_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None):
        """
        Initialize broker API error.
        
        Args:
            message: Error message
            api_code: Broker-specific error code
            context: Additional context information
        """
        super().__init__(message, context)
        self.api_code = api_code


class BrokerRateLimitError(BrokerError):
    """Raised when broker rate limit is exceeded."""
    pass


class InsufficientFundsError(BrokerError):
    """Raised when account has insufficient funds."""
    pass


class InvalidOrderError(BrokerError):
    """Raised when order parameters are invalid."""
    pass


# Feature engineering exceptions
class FeatureEngineeringError(TradingSystemError):
    """Base class for feature engineering errors."""
    pass


class IndicatorCalculationError(FeatureEngineeringError):
    """Raised when indicator calculation fails."""
    pass


class InsufficientDataError(FeatureEngineeringError):
    """Raised when insufficient data for calculation."""
    pass


class InvalidParameterError(FeatureEngineeringError):
    """Raised when indicator parameters are invalid."""
    pass


class FeaturePipelineError(FeatureEngineeringError):
    """Raised when feature pipeline execution fails."""
    pass


# Configuration exceptions
class ConfigurationError(TradingSystemError):
    """Base class for configuration errors."""
    pass


class InvalidConfigError(ConfigurationError):
    """Raised when configuration is invalid."""
    pass


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing."""
    pass


class ConfigParsingError(ConfigurationError):
    """Raised when configuration parsing fails."""
    pass


# Strategy exceptions
class StrategyError(TradingSystemError):
    """Base class for strategy-related errors."""
    pass


class StrategyInitializationError(StrategyError):
    """Raised when strategy initialization fails."""
    pass


class SignalGenerationError(StrategyError):
    """Raised when signal generation fails."""
    pass


class RiskManagementError(StrategyError):
    """Raised when risk management checks fail."""
    pass


class PositionSizingError(StrategyError):
    """Raised when position sizing calculation fails."""
    pass


# Trading execution exceptions
class TradingExecutionError(TradingSystemError):
    """Base class for trading execution errors."""
    pass


class OrderExecutionError(TradingExecutionError):
    """Raised when order execution fails."""
    pass


class OrderRejectionError(TradingExecutionError):
    """Raised when order is rejected."""
    pass


class SlippageError(TradingExecutionError):
    """Raised when slippage exceeds acceptable limits."""
    pass


class LiquidityError(TradingExecutionError):
    """Raised when insufficient liquidity for order."""
    pass


# Calendar and time exceptions
class CalendarError(TradingSystemError):
    """Base class for calendar-related errors."""
    pass


class MarketClosedError(CalendarError):
    """Raised when attempting to trade during market closure."""
    pass


class InvalidTimeframeError(CalendarError):
    """Raised when timeframe is invalid."""
    pass


class TimezoneError(CalendarError):
    """Raised when timezone handling fails."""
    pass


# Symbol and mapping exceptions
class SymbolError(TradingSystemError):
    """Base class for symbol-related errors."""
    pass


class InvalidSymbolError(SymbolError):
    """Raised when symbol is invalid or not supported."""
    pass


class SymbolMappingError(SymbolError):
    """Raised when symbol mapping fails."""
    pass


class UnsupportedAssetClassError(SymbolError):
    """Raised when asset class is not supported."""
    pass


# Processing and computation exceptions
class ProcessingError(TradingSystemError):
    """Base class for processing errors."""
    pass


class CalculationError(ProcessingError):
    """Raised when mathematical calculation fails."""
    pass


class MemoryError(ProcessingError):
    """Raised when memory constraints are exceeded."""
    pass


class TimeoutError(ProcessingError):
    """Raised when operation times out."""
    pass


class ConcurrencyError(ProcessingError):
    """Raised when concurrency issues occur."""
    pass


# Factory and registry exceptions
class FactoryError(TradingSystemError):
    """Base class for factory-related errors."""
    pass


class UnregisteredTypeError(FactoryError):
    """Raised when attempting to create unregistered type."""
    pass


class RegistrationError(FactoryError):
    """Raised when type registration fails."""
    pass


class CreationError(FactoryError):
    """Raised when object creation fails."""
    pass


# Utility functions for exception handling
def format_validation_error(field_name: str, value: Any, expected: str) -> str:
    """
    Format a validation error message.
    
    Args:
        field_name: Name of the field that failed validation
        value: The invalid value
        expected: Description of expected value
        
    Returns:
        Formatted error message
    """
    return f"Invalid {field_name}: got {value}, expected {expected}"


def create_context(**kwargs) -> Dict[str, Any]:
    """
    Create context dictionary for exceptions.
    
    Args:
        **kwargs: Key-value pairs for context
        
    Returns:
        Context dictionary
    """
    return {k: v for k, v in kwargs.items() if v is not None} 