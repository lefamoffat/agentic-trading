"""Tests for market data exceptions."""

import pytest

from src.market_data.exceptions import (
    MarketDataError,
    DataSourceError,
    DataRangeError,
    StorageError,
    SymbolError,
    TimeframeError
)
from src.utils.exceptions import DataError


@pytest.mark.unit
class TestMarketDataExceptions:
    """Test market data exception hierarchy."""
    
    def test_market_data_error_inheritance(self):
        """Test that MarketDataError inherits from DataError."""
        error = MarketDataError("Test error")
        assert isinstance(error, DataError)
        assert isinstance(error, Exception)
    
    def test_data_source_error_inheritance(self):
        """Test that DataSourceError inherits from MarketDataError."""
        error = DataSourceError("Broker connection failed")
        assert isinstance(error, MarketDataError)
        assert isinstance(error, DataError)
    
    def test_data_range_error_inheritance(self):
        """Test that DataRangeError inherits from MarketDataError."""
        error = DataRangeError("Invalid date range")
        assert isinstance(error, MarketDataError)
        assert isinstance(error, DataError)
    
    def test_storage_error_inheritance(self):
        """Test that StorageError inherits from MarketDataError."""
        error = StorageError("Cache write failed")
        assert isinstance(error, MarketDataError)
        assert isinstance(error, DataError)
    
    def test_symbol_error_inheritance(self):
        """Test that SymbolError inherits from MarketDataError."""
        error = SymbolError("Unsupported symbol")
        assert isinstance(error, MarketDataError)
        assert isinstance(error, DataError)
    
    def test_timeframe_error_inheritance(self):
        """Test that TimeframeError inherits from MarketDataError."""
        error = TimeframeError("Invalid timeframe")
        assert isinstance(error, MarketDataError)
        assert isinstance(error, DataError)
    
    def test_exception_messages(self):
        """Test that exception messages are preserved."""
        message = "Test error message"
        
        errors = [
            MarketDataError(message),
            DataSourceError(message),
            DataRangeError(message),
            StorageError(message),
            SymbolError(message),
            TimeframeError(message)
        ]
        
        for error in errors:
            assert str(error) == message
    
    def test_exception_context(self):
        """Test that exceptions can carry context (inherited from base)."""
        context = {"broker": "forex_com", "symbol": "EUR/USD"}
        error = DataSourceError("Connection failed", context)
        
        assert error.context == context
        assert "context=" in str(error) 