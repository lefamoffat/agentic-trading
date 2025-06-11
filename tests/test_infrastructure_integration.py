"""
Integration tests for the project infrastructure.
Tests that types, exceptions, and validation work together properly.
"""

import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.types import (
    BrokerType, IndicatorType, Timeframe, OrderType, OrderSide,
    SymbolType, OHLCVData, ValidationResult, ErrorContext
)

from src.exceptions import (
    TradingSystemError, ValidationError, IndicatorCalculationError,
    BrokerAPIError, InvalidParametersError, format_error_message,
    create_error_context
)

from src.utils.validation import (
    validate_ohlcv_data, validate_indicator_parameters,
    validate_positive_integer, validate_enum_value,
    calculate_data_quality_score
)


class TestTypesExceptionsIntegration:
    """Test integration between types and exceptions."""
    
    def test_enum_validation_with_custom_exceptions(self):
        """Test that enum validation works with custom exceptions."""
        # Valid enum validation
        broker = validate_enum_value(BrokerType.FOREX_COM, BrokerType)
        assert broker == BrokerType.FOREX_COM
        
        # Invalid enum should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            validate_enum_value("invalid_broker", BrokerType)
        
        assert isinstance(exc_info.value, TradingSystemError)
        assert "must be one of" in str(exc_info.value)
    
    def test_indicator_type_with_parameter_validation(self):
        """Test indicator types with parameter validation."""
        # Test valid indicator with valid parameters
        params = {'period': 20}
        validated_params = validate_indicator_parameters(IndicatorType.SMA, params)
        assert validated_params['period'] == 20
        
        # Test invalid parameters raise appropriate exception
        with pytest.raises(ValidationError) as exc_info:
            validate_indicator_parameters(IndicatorType.RSI, {'period': -5})
        
        # Should be our custom exception with context
        assert isinstance(exc_info.value, TradingSystemError)
        assert "period must be positive" in str(exc_info.value)
    
    def test_timeframe_validation_integration(self):
        """Test timeframe validation integration."""
        # Valid timeframe
        tf = validate_enum_value(Timeframe.H1, Timeframe)
        assert tf == Timeframe.H1
        assert tf.minutes == 60
        
        # Test with string value
        tf_from_string = validate_enum_value("1h", Timeframe)
        assert tf_from_string == Timeframe.H1
        
        # Invalid timeframe
        with pytest.raises(ValidationError):
            validate_enum_value("invalid_timeframe", Timeframe)
    
    def test_symbol_type_with_context(self):
        """Test symbol type usage with error context."""
        symbol: SymbolType = "EUR/USD"
        
        # Create error context using the symbol
        context = create_error_context(
            operation="data_fetch",
            symbol=symbol,
            broker=BrokerType.FOREX_COM.value,
            timeframe=Timeframe.H1.value
        )
        
        # Create error with context
        error = BrokerAPIError("Failed to fetch data", context=context)
        
        assert error.context['symbol'] == "EUR/USD"
        assert error.context['broker'] == "forex_com"
        assert error.context['timeframe'] == "1h"


class TestValidationExceptionsIntegration:
    """Test integration between validation and exceptions."""
    
    def create_sample_ohlcv(self):
        """Create sample OHLCV data for testing."""
        return pd.DataFrame({
            'open': [1.0980, 1.0985, 1.0990],
            'high': [1.0995, 1.1000, 1.1005],
            'low': [1.0975, 1.0980, 1.0985],
            'close': [1.0985, 1.0990, 1.0995],
            'volume': [1000, 1500, 1200]
        })
    
    def test_ohlcv_validation_with_custom_exceptions(self):
        """Test OHLCV validation with custom exception handling."""
        # Valid data should pass
        valid_data = self.create_sample_ohlcv()
        result = validate_ohlcv_data(valid_data)
        assert isinstance(result, pd.DataFrame)
        
        # Invalid data should raise ValidationError
        invalid_data = pd.DataFrame({
            'open': [1.0980, 1.0985],
            'high': [1.0995, 1.1000]
            # Missing required columns
        })
        
        with pytest.raises(ValidationError) as exc_info:
            validate_ohlcv_data(invalid_data)
        
        assert isinstance(exc_info.value, TradingSystemError)
        assert "Missing required OHLCV columns" in str(exc_info.value)
    
    def test_validation_result_type_usage(self):
        """Test using ValidationResult type with actual validation."""
        data = self.create_sample_ohlcv()
        
        # Simulate a validation function returning ValidationResult
        def validate_data_custom(df: pd.DataFrame) -> ValidationResult:
            try:
                validate_ohlcv_data(df)
                return {
                    'is_valid': True,
                    'errors': [],
                    'warnings': []
                }
            except ValidationError as e:
                return {
                    'is_valid': False,
                    'errors': [str(e)],
                    'warnings': []
                }
        
        # Test with valid data
        result = validate_data_custom(data)
        assert result['is_valid'] is True
        assert len(result['errors']) == 0
        
        # Test with invalid data
        invalid_data = pd.DataFrame({'invalid': [1, 2, 3]})
        result = validate_data_custom(invalid_data)
        assert result['is_valid'] is False
        assert len(result['errors']) > 0
    
    def test_error_context_with_validation_failure(self):
        """Test error context creation during validation failure."""
        symbol = "EUR/USD"
        timeframe = Timeframe.H1
        
        try:
            # Simulate validation failure
            invalid_period = -5
            validate_positive_integer(invalid_period, "period")
        except ValidationError as e:
            # Create enriched error with context
            context = create_error_context(
                operation="indicator_validation",
                symbol=symbol,
                timeframe=timeframe.value,
                indicator=IndicatorType.RSI.value,
                parameter="period",
                value=invalid_period
            )
            
            # Raise new error with context
            raise IndicatorCalculationError(
                "Indicator parameter validation failed", 
                context=context
            ) from e
        
        # This should not be reached in normal flow
        pytest.fail("Expected IndicatorCalculationError to be raised")


class TestRealWorldScenarios:
    """Test real-world scenarios using the complete infrastructure."""
    
    def test_broker_data_fetch_simulation(self):
        """Simulate a complete broker data fetch with error handling."""
        def simulate_broker_fetch(
            broker: BrokerType,
            symbol: SymbolType,
            timeframe: Timeframe
        ) -> OHLCVData:
            """Simulate broker data fetch with proper error handling."""
            
            # Validate inputs using our validation system
            broker = validate_enum_value(broker, BrokerType)
            timeframe = validate_enum_value(timeframe, Timeframe)
            
            # Simulate some validation failure scenarios
            if symbol == "INVALID/PAIR":
                context = create_error_context(
                    operation="data_fetch",
                    broker=broker.value,
                    symbol=symbol,
                    timeframe=timeframe.value
                )
                raise BrokerAPIError("Unsupported symbol", context=context)
            
            # Return mock data for valid scenarios
            return pd.DataFrame({
                'open': [1.0980, 1.0985],
                'high': [1.0995, 1.1000],
                'low': [1.0975, 1.0980],
                'close': [1.0985, 1.0990],
                'volume': [1000, 1500]
            })
        
        # Test successful fetch
        data = simulate_broker_fetch(
            BrokerType.FOREX_COM,
            "EUR/USD",
            Timeframe.H1
        )
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 2
        
        # Test error scenario
        with pytest.raises(BrokerAPIError) as exc_info:
            simulate_broker_fetch(
                BrokerType.OANDA,
                "INVALID/PAIR",
                Timeframe.M5
            )
        
        error = exc_info.value
        assert error.context['operation'] == "data_fetch"
        assert error.context['symbol'] == "INVALID/PAIR"
        assert error.context['broker'] == "oanda"
        assert error.context['timeframe'] == "5m"
    
    def test_indicator_calculation_pipeline(self):
        """Test a complete indicator calculation pipeline."""
        def calculate_indicator_safe(
            indicator_type: IndicatorType,
            data: OHLCVData,
            **params
        ) -> pd.DataFrame:
            """Safely calculate indicator with comprehensive error handling."""
            
            try:
                # Validate indicator type
                indicator_type = validate_enum_value(indicator_type, IndicatorType)
                
                # Validate data
                data = validate_ohlcv_data(data)
                
                # Validate parameters
                validated_params = validate_indicator_parameters(indicator_type, params)
                
                # Simulate indicator calculation
                if indicator_type == IndicatorType.SMA:
                    period = validated_params['period']
                    if len(data) < period:
                        context = create_error_context(
                            operation="sma_calculation",
                            indicator=indicator_type.value,
                            required_periods=period,
                            available_periods=len(data)
                        )
                        raise InsufficientDataError(
                            f"Need at least {period} periods for SMA calculation",
                            context=context
                        )
                    
                    # Simple SMA calculation
                    sma_values = data['close'].rolling(window=period).mean()
                    return pd.DataFrame({'sma': sma_values})
                
                # For other indicators, return dummy data
                return pd.DataFrame({'value': [0.0] * len(data)})
                
            except ValidationError as e:
                # Re-raise as indicator calculation error with context
                context = create_error_context(
                    operation="indicator_calculation",
                    indicator=indicator_type.value if isinstance(indicator_type, IndicatorType) else str(indicator_type),
                    data_length=len(data) if isinstance(data, pd.DataFrame) else 0
                )
                raise IndicatorCalculationError(
                    f"Indicator calculation failed: {e}",
                    context=context
                ) from e
        
        # Test successful calculation
        data = pd.DataFrame({
            'open': [1.0] * 25,
            'high': [1.1] * 25,
            'low': [0.9] * 25,
            'close': [1.05] * 25,
            'volume': [1000] * 25
        })
        
        result = calculate_indicator_safe(IndicatorType.SMA, data, period=20)
        assert isinstance(result, pd.DataFrame)
        assert 'sma' in result.columns
        
        # Test with insufficient data
        short_data = data.head(10)  # Only 10 periods
        
        with pytest.raises(IndicatorCalculationError) as exc_info:
            calculate_indicator_safe(IndicatorType.SMA, short_data, period=20)
        
        error = exc_info.value
        assert error.context['operation'] == "indicator_calculation"
        assert error.context['indicator'] == "sma"
        assert "insufficient data" in str(error).lower()
        
        # Test with invalid parameters
        with pytest.raises(IndicatorCalculationError) as exc_info:
            calculate_indicator_safe(IndicatorType.SMA, data, period=-5)
        
        # Should be wrapped IndicatorCalculationError
        assert isinstance(exc_info.value, IndicatorCalculationError)
        assert exc_info.value.__cause__ is not None  # Should have original cause
    
    def test_error_message_formatting_integration(self):
        """Test error message formatting with real scenarios."""
        # Test comprehensive error message formatting
        context = create_error_context(
            operation="backtest_execution",
            strategy="MeanReversion",
            symbol="GBP/USD",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-12-31",
            error_step="signal_generation"
        )
        
        error_msg = format_error_message(
            "Strategy execution failed at signal generation step",
            "BacktestEngine",
            context
        )
        
        # Verify comprehensive error message
        assert "Strategy execution failed" in error_msg
        assert "BacktestEngine" in error_msg
        assert "MeanReversion" in error_msg
        assert "GBP/USD" in error_msg
        assert "signal_generation" in error_msg
    
    def test_data_quality_validation_pipeline(self):
        """Test complete data quality validation pipeline."""
        def assess_data_quality(data: pd.DataFrame, symbol: str) -> ValidationResult:
            """Assess data quality with comprehensive validation."""
            errors = []
            warnings = []
            
            try:
                # Basic OHLCV validation
                data = validate_ohlcv_data(data)
                
                # Calculate quality score
                quality_score = calculate_data_quality_score(data)
                
                if quality_score < 0.8:
                    warnings.append(f"Data quality score is low: {quality_score:.2f}")
                
                if quality_score < 0.5:
                    errors.append("Data quality is too poor for reliable analysis")
                
                # Check data length
                if len(data) < 100:
                    warnings.append(f"Limited data available: only {len(data)} periods")
                
                if len(data) < 20:
                    errors.append("Insufficient data for analysis")
                
                return {
                    'is_valid': len(errors) == 0,
                    'errors': errors,
                    'warnings': warnings
                }
                
            except ValidationError as e:
                return {
                    'is_valid': False,
                    'errors': [str(e)],
                    'warnings': warnings
                }
        
        # Test with good quality data
        good_data = pd.DataFrame({
            'open': np.random.normal(1.1, 0.01, 200),
            'high': np.random.normal(1.11, 0.01, 200),
            'low': np.random.normal(1.09, 0.01, 200),
            'close': np.random.normal(1.1, 0.01, 200),
            'volume': np.random.randint(1000, 5000, 200)
        })
        
        result = assess_data_quality(good_data, "EUR/USD")
        assert result['is_valid'] is True
        assert len(result['errors']) == 0
        
        # Test with poor quality data (many NaNs)
        poor_data = good_data.copy()
        poor_data.loc[poor_data.index % 2 == 0, 'close'] = np.nan  # 50% missing
        
        result = assess_data_quality(poor_data, "EUR/USD")
        assert result['is_valid'] is False
        assert len(result['errors']) > 0
        
        # Test with insufficient data
        small_data = good_data.head(10)
        result = assess_data_quality(small_data, "EUR/USD")
        assert len(result['warnings']) > 0  # Should have warnings about data size


class TestErrorRecoveryScenarios:
    """Test error recovery and graceful degradation scenarios."""
    
    def test_graceful_degradation_with_partial_data(self):
        """Test graceful degradation when some data is missing."""
        def process_with_fallback(data: pd.DataFrame) -> dict:
            """Process data with fallback strategies."""
            results = {}
            
            try:
                # Try full validation first
                validate_ohlcv_data(data)
                results['status'] = 'full_validation_passed'
                results['quality_score'] = calculate_data_quality_score(data)
                
            except ValidationError as e:
                # Try partial processing
                results['status'] = 'partial_processing'
                results['validation_error'] = str(e)
                
                # Check what we can salvage
                if 'close' in data.columns and data['close'].notna().any():
                    results['close_data_available'] = True
                    results['close_data_points'] = data['close'].notna().sum()
                else:
                    results['close_data_available'] = False
                
                if 'volume' in data.columns and data['volume'].notna().any():
                    results['volume_data_available'] = True
                else:
                    results['volume_data_available'] = False
            
            return results
        
        # Test with complete data
        complete_data = pd.DataFrame({
            'open': [1.0, 1.1, 1.2],
            'high': [1.1, 1.2, 1.3],
            'low': [0.9, 1.0, 1.1],
            'close': [1.05, 1.15, 1.25],
            'volume': [1000, 1100, 1200]
        })
        
        result = process_with_fallback(complete_data)
        assert result['status'] == 'full_validation_passed'
        assert result['quality_score'] == 1.0
        
        # Test with partial data
        partial_data = pd.DataFrame({
            'close': [1.05, 1.15, 1.25],
            'volume': [1000, 1100, 1200]
            # Missing OHLC columns
        })
        
        result = process_with_fallback(partial_data)
        assert result['status'] == 'partial_processing'
        assert result['close_data_available'] is True
        assert result['volume_data_available'] is True
        assert 'validation_error' in result
    
    def test_exception_chaining_preservation(self):
        """Test that exception chaining preserves full error context."""
        def multi_level_operation():
            """Simulate multi-level operation with exception chaining."""
            try:
                # Level 1: Parameter validation
                try:
                    validate_positive_integer(-5, "period")
                except ValidationError as e:
                    context = create_error_context(
                        level="parameter_validation",
                        parameter="period",
                        value=-5
                    )
                    raise InvalidParametersError(
                        "Parameter validation failed",
                        context=context
                    ) from e
                
            except InvalidParametersError as e:
                # Level 2: Indicator calculation
                context = create_error_context(
                    level="indicator_calculation",
                    indicator="RSI",
                    operation="calculate_rsi"
                )
                context.update(e.context or {})  # Preserve previous context
                
                raise IndicatorCalculationError(
                    "RSI calculation failed due to invalid parameters",
                    context=context
                ) from e
        
        with pytest.raises(IndicatorCalculationError) as exc_info:
            multi_level_operation()
        
        final_error = exc_info.value
        
        # Check that we have the full chain
        assert isinstance(final_error, IndicatorCalculationError)
        assert isinstance(final_error.__cause__, InvalidParametersError)
        assert isinstance(final_error.__cause__.__cause__, ValidationError)
        
        # Check that context is preserved and combined
        assert final_error.context['level'] == "indicator_calculation"
        assert final_error.context['indicator'] == "RSI"
        assert final_error.context['parameter'] == "period"
        assert final_error.context['value'] == -5


if __name__ == "__main__":
    pytest.main([__file__]) 