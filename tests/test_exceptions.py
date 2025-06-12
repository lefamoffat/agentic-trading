"""
Unit tests for the project-wide exception system.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.exceptions import (
    # Base exception
    TradingSystemError,
    
    # Data errors
    ValidationError, DataQualityError, MissingDataError, 
    InvalidTimeframeError, DataIntegrityError,
    
    # Broker errors
    BrokerAPIError, BrokerConnectionError, BrokerAuthenticationError,
    InsufficientFundsError, OrderRejectedError, BrokerRateLimitError,
    MarketClosedError,
    
    # Feature engineering errors
    IndicatorCalculationError, FeaturePipelineError, 
    InvalidParameterError, InsufficientDataError,
    
    # Configuration errors
    InvalidConfigError, MissingConfigError, ConfigurationError,
    
    # Strategy errors
    SignalGenerationError, RiskManagementError, StrategyError,
    BacktestError, OptimizationError,
    
    # Trading execution errors
    OrderExecutionError, SlippageError, PositionError,
    
    # Calendar errors
    CalendarError, EventDataError,
    
    # Symbol and processing errors
    SymbolError, ProcessingError,
    
    # Factory errors
    FactoryError,
    
    # Utility functions
    format_validation_error, create_context
)


class TestBaseTradingSystemError:
    """Test the base TradingSystemError class."""
    
    def test_basic_creation(self):
        """Test basic error creation."""
        error = TradingSystemError("Test error message")
        assert str(error) == "Test error message"
        assert error.context is None
    
    def test_creation_with_context(self):
        """Test error creation with context."""
        context = {"operation": "test", "symbol": "EUR/USD"}
        error = TradingSystemError("Test error", context=context)
        
        assert str(error) == "Test error"
        assert error.context == context
        assert error.context["operation"] == "test"
        assert error.context["symbol"] == "EUR/USD"
    
    def test_inheritance(self):
        """Test that TradingSystemError inherits from Exception."""
        error = TradingSystemError("Test")
        assert isinstance(error, Exception)
        assert isinstance(error, TradingSystemError)
    
    def test_context_attribute_access(self):
        """Test accessing context attributes."""
        context = {"symbol": "GBP/USD", "timeframe": "1h", "period": 14}
        error = TradingSystemError("Test", context=context)
        
        # Context should be accessible as a dictionary
        assert error.context["symbol"] == "GBP/USD"
        assert error.context["timeframe"] == "1h"
        assert error.context["period"] == 14


class TestDataErrors:
    """Test data-related exception classes."""
    
    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("Invalid data format")
        assert isinstance(error, TradingSystemError)
        assert str(error) == "Invalid data format"
    
    def test_validation_error_with_context(self):
        """Test ValidationError with context."""
        context = {"column": "close", "expected_type": "float", "actual_type": "str"}
        error = ValidationError("Column type mismatch", context=context)
        
        assert error.context["column"] == "close"
        assert error.context["expected_type"] == "float"
        assert error.context["actual_type"] == "str"
    
    def test_data_quality_error(self):
        """Test DataQualityError."""
        error = DataQualityError("Data contains too many gaps")
        assert isinstance(error, TradingSystemError)
        assert str(error) == "Data contains too many gaps"
    
    def test_missing_data_error(self):
        """Test MissingDataError."""
        error = MissingDataError("Required column 'volume' not found")
        assert isinstance(error, TradingSystemError)
    
    def test_invalid_timeframe_error(self):
        """Test InvalidTimeframeError."""
        error = InvalidTimeframeError("Unsupported timeframe: 3m")
        assert isinstance(error, TradingSystemError)
    
    def test_data_integrity_error(self):
        """Test DataIntegrityError."""
        error = DataIntegrityError("OHLC data consistency check failed")
        assert isinstance(error, TradingSystemError)


class TestBrokerErrors:
    """Test broker-related exception classes."""
    
    def test_broker_api_error(self):
        """Test BrokerAPIError."""
        error = BrokerAPIError("API returned status 500")
        assert isinstance(error, TradingSystemError)
        assert str(error) == "API returned status 500"
    
    def test_broker_api_error_with_status_code(self):
        """Test BrokerAPIError with status code context."""
        context = {"status_code": 500, "endpoint": "/api/orders", "response": "Internal Server Error"}
        error = BrokerAPIError("API request failed", context=context)
        
        assert error.context["status_code"] == 500
        assert error.context["endpoint"] == "/api/orders"
    
    def test_broker_connection_error(self):
        """Test BrokerConnectionError."""
        error = BrokerConnectionError("Failed to connect to broker")
        assert isinstance(error, TradingSystemError)
    
    def test_authentication_error(self):
        """Test BrokerAuthenticationError."""
        error = BrokerAuthenticationError("Invalid credentials")
        assert isinstance(error, TradingSystemError)
    
    def test_insufficient_funds_error(self):
        """Test InsufficientFundsError."""
        context = {"required": 10000.0, "available": 5000.0, "currency": "USD"}
        error = InsufficientFundsError("Not enough funds for trade", context=context)
        
        assert error.context["required"] == 10000.0
        assert error.context["available"] == 5000.0
    
    def test_order_rejected_error(self):
        """Test OrderRejectedError."""
        error = OrderRejectedError("Order rejected by broker")
        assert isinstance(error, TradingSystemError)
    
    def test_rate_limit_error(self):
        """Test BrokerRateLimitError."""
        context = {"requests_per_minute": 100, "retry_after": 60}
        error = BrokerRateLimitError("Rate limit exceeded", context=context)
        
        assert error.context["requests_per_minute"] == 100
        assert error.context["retry_after"] == 60
    
    def test_market_closed_error(self):
        """Test MarketClosedError."""
        error = MarketClosedError("Market is closed for trading")
        assert isinstance(error, TradingSystemError)


class TestFeatureEngineeringErrors:
    """Test feature engineering exception classes."""
    
    def test_indicator_calculation_error(self):
        """Test IndicatorCalculationError."""
        context = {"indicator": "RSI", "period": 14, "data_length": 10}
        error = IndicatorCalculationError("Insufficient data for RSI calculation", context=context)
        
        assert error.context["indicator"] == "RSI"
        assert error.context["period"] == 14
        assert error.context["data_length"] == 10
    
    def test_feature_pipeline_error(self):
        """Test FeaturePipelineError."""
        error = FeaturePipelineError("Pipeline step failed")
        assert isinstance(error, TradingSystemError)
    
    def test_invalid_parameter_error(self):
        """Test InvalidParameterError."""
        context = {"parameter": "period", "value": -5, "constraint": "must be positive"}
        error = InvalidParameterError("Invalid parameter value", context=context)
        
        assert error.context["parameter"] == "period"
        assert error.context["value"] == -5
    
    def test_insufficient_data_error(self):
        """Test InsufficientDataError."""
        error = InsufficientDataError("Need at least 20 data points")
        assert isinstance(error, TradingSystemError)


class TestConfigurationErrors:
    """Test configuration exception classes."""
    
    def test_invalid_config_error(self):
        """Test InvalidConfigError."""
        error = InvalidConfigError("Invalid configuration parameter")
        assert isinstance(error, TradingSystemError)
    
    def test_missing_config_error(self):
        """Test MissingConfigError."""
        context = {"required_keys": ["api_key", "api_secret"], "section": "broker"}
        error = MissingConfigError("Required configuration keys missing", context=context)
        
        assert error.context["required_keys"] == ["api_key", "api_secret"]
        assert error.context["section"] == "broker"
    
    def test_configuration_error(self):
        """Test generic ConfigurationError."""
        error = ConfigurationError("Configuration validation failed")
        assert isinstance(error, TradingSystemError)


class TestStrategyErrors:
    """Test strategy-related exception classes."""
    
    def test_signal_generation_error(self):
        """Test SignalGenerationError."""
        context = {"strategy": "MeanReversion", "symbol": "EUR/USD", "step": "signal_calculation"}
        error = SignalGenerationError("Failed to generate trading signal", context=context)
        
        assert error.context["strategy"] == "MeanReversion"
        assert error.context["symbol"] == "EUR/USD"
    
    def test_risk_management_error(self):
        """Test RiskManagementError."""
        error = RiskManagementError("Position size exceeds risk limits")
        assert isinstance(error, TradingSystemError)
    
    def test_strategy_error(self):
        """Test generic StrategyError."""
        error = StrategyError("Strategy execution failed")
        assert isinstance(error, TradingSystemError)
    
    def test_backtest_error(self):
        """Test BacktestError."""
        error = BacktestError("Backtest simulation failed")
        assert isinstance(error, TradingSystemError)
    
    def test_optimization_error(self):
        """Test OptimizationError."""
        error = OptimizationError("Parameter optimization failed")
        assert isinstance(error, TradingSystemError)


class TestTradingExecutionErrors:
    """Test trading execution exception classes."""
    
    def test_order_execution_error(self):
        """Test OrderExecutionError."""
        context = {"order_id": "12345", "symbol": "GBP/USD", "side": "buy", "quantity": 10000}
        error = OrderExecutionError("Order execution failed", context=context)
        
        assert error.context["order_id"] == "12345"
        assert error.context["symbol"] == "GBP/USD"
    
    def test_slippage_error(self):
        """Test SlippageError."""
        context = {"expected_price": 1.2500, "executed_price": 1.2505, "slippage": 0.0005}
        error = SlippageError("Excessive slippage detected", context=context)
        
        assert error.context["expected_price"] == 1.2500
        assert error.context["executed_price"] == 1.2505
    
    def test_position_error(self):
        """Test PositionError."""
        error = PositionError("Position management error")
        assert isinstance(error, TradingSystemError)


class TestSpecializedErrors:
    """Test other specialized exception classes."""
    
    def test_calendar_error(self):
        """Test CalendarError."""
        error = CalendarError("Failed to load economic calendar")
        assert isinstance(error, TradingSystemError)
    
    def test_event_data_error(self):
        """Test EventDataError."""
        error = EventDataError("Economic event data parsing failed")
        assert isinstance(error, TradingSystemError)
    
    def test_symbol_error(self):
        """Test SymbolError."""
        context = {"symbol": "INVALID/PAIR", "broker": "forex_com"}
        error = SymbolError("Unsupported symbol", context=context)
        
        assert error.context["symbol"] == "INVALID/PAIR"
        assert error.context["broker"] == "forex_com"
    
    def test_processing_error(self):
        """Test ProcessingError."""
        error = ProcessingError("Data processing failed")
        assert isinstance(error, TradingSystemError)
    
    def test_factory_error(self):
        """Test FactoryError."""
        context = {"component_type": "unknown_indicator", "available_types": ["sma", "ema", "rsi"]}
        error = FactoryError("Unknown component type", context=context)
        
        assert error.context["component_type"] == "unknown_indicator"
        assert error.context["available_types"] == ["sma", "ema", "rsi"]


class TestUtilityFunctions:
    """Test utility functions for error handling."""
    
    def test_format_validation_error_basic(self):
        """Test basic validation error formatting."""
        message = format_validation_error("period", -5, "positive integer")
        assert "period" in message
        assert str(-5) in message
        assert "positive integer" in message
    
    def test_format_validation_error_detailed(self):
        """Test detailed validation error formatting."""
        message = format_validation_error("timeframe", "invalid", "one of: 1m, 5m, 15m")
        
        assert "timeframe" in message
        assert "invalid" in message
        assert "one of" in message
    
    def test_create_context_basic(self):
        """Test basic context creation."""
        context = create_context(operation="test", symbol="EUR/USD")
        assert context["operation"] == "test"
        assert context["symbol"] == "EUR/USD"
    
    def test_create_context_detailed(self):
        """Test creating detailed context dictionary."""
        context = create_context(
            operation="indicator_calculation",
            symbol="GBP/USD",
            custom_field="custom_value"
        )
        
        assert context["operation"] == "indicator_calculation"
        assert context["symbol"] == "GBP/USD"
        assert context["custom_field"] == "custom_value"
    
    def test_create_context_empty(self):
        """Test creating empty context."""
        context = create_context()
        assert isinstance(context, dict)
        assert len(context) == 0
    
    def test_create_context_overwrites(self):
        """Test that create_context handles keyword arguments properly."""
        # Test normal context creation
        context = create_context(
            operation="test_operation",
            symbol="EUR/USD"
        )
        
        # Should have both values
        assert context["operation"] == "test_operation"
        assert context["symbol"] == "EUR/USD"


class TestExceptionInheritance:
    """Test exception inheritance hierarchy."""
    
    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from TradingSystemError."""
        exception_classes = [
            ValidationError, DataQualityError, MissingDataError,
            BrokerAPIError, BrokerConnectionError, BrokerAuthenticationError,
            IndicatorCalculationError, FeaturePipelineError,
            InvalidConfigError, SignalGenerationError,
            OrderExecutionError, CalendarError, SymbolError,
            ProcessingError, FactoryError
        ]
        
        for exc_class in exception_classes:
            # Create instance and check inheritance
            instance = exc_class("Test message")
            assert isinstance(instance, TradingSystemError)
            assert isinstance(instance, Exception)
    
    def test_exception_categories(self):
        """Test that exceptions are properly categorized."""
        # Data-related errors
        data_errors = [ValidationError, DataQualityError, MissingDataError]
        for exc_class in data_errors:
            instance = exc_class("Test")
            assert isinstance(instance, TradingSystemError)
        
        # Broker-related errors
        broker_errors = [BrokerAPIError, BrokerConnectionError, BrokerAuthenticationError]
        for exc_class in broker_errors:
            instance = exc_class("Test")
            assert isinstance(instance, TradingSystemError)
        
        # Feature engineering errors
        feature_errors = [IndicatorCalculationError, FeaturePipelineError]
        for exc_class in feature_errors:
            instance = exc_class("Test")
            assert isinstance(instance, TradingSystemError)


class TestExceptionChaining:
    """Test exception chaining and context preservation."""
    
    def test_exception_chaining_with_cause(self):
        """Test exception chaining using 'from' clause."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise IndicatorCalculationError("Calculation failed") from e
        except IndicatorCalculationError as final_error:
            assert isinstance(final_error.__cause__, ValueError)
            assert str(final_error.__cause__) == "Original error"
            assert str(final_error) == "Calculation failed"
    
    def test_nested_context_preservation(self):
        """Test that context is preserved through exception chains."""
        original_context = {"indicator": "RSI", "period": 14}
        
        try:
            try:
                raise ValidationError("Data validation failed", context=original_context)
            except ValidationError as e:
                # Create new exception with additional context
                new_context = e.context.copy() if e.context else {}
                new_context.update({"symbol": "EUR/USD", "timeframe": "1h"})
                raise IndicatorCalculationError("Calculation failed", context=new_context) from e
        except IndicatorCalculationError as final_error:
            # Should have combined context
            assert final_error.context["indicator"] == "RSI"
            assert final_error.context["period"] == 14
            assert final_error.context["symbol"] == "EUR/USD"
            assert final_error.context["timeframe"] == "1h"


if __name__ == "__main__":
    pytest.main([__file__]) 