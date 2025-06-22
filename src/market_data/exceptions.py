"""Market data specific exceptions.

These exceptions extend the existing exception hierarchy from src.utils.exceptions
to provide specific error handling for market data operations.
"""

from src.utils.exceptions import ConfigurationError, DataError

class MarketDataError(DataError):
    """Base exception for market data operations."""
    pass

class DataSourceError(MarketDataError):
    """Exception for data source related errors (broker failures, API errors, etc.)."""
    pass

class DataRangeError(MarketDataError):
    """Exception for invalid date ranges or time-related errors."""
    pass

class StorageError(MarketDataError):
    """Exception for storage operations (caching, qlib conversion, file I/O)."""
    pass

class SymbolError(MarketDataError):
    """Exception for symbol-related errors (unsupported symbols, invalid formats)."""
    pass

class TimeframeError(MarketDataError):
    """Exception for timeframe-related errors (unsupported timeframes, conversion issues)."""
    pass 