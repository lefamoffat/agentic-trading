"""
Unit tests for the validation utilities.
"""

import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.validation import (
    # Parameter validation functions
    validate_positive_number, validate_non_negative_number,
    validate_positive_integer, validate_range,
    validate_dataframe_columns, validate_series_numeric,
    
    # OHLCV validation
    validate_ohlcv_data, validate_ohlcv_consistency,
    validate_data_completeness, validate_data_quality,
    
    # Indicator parameter validation
    validate_indicator_parameters, create_parameter_schema,
    validate_against_schema,
    
    # Type validation
    validate_type, validate_enum_value,
    
    # Utility functions
    check_data_gaps, calculate_data_quality_score
)

from src.exceptions import ValidationError
from src.types import IndicatorType, Timeframe


class TestParameterValidation:
    """Test basic parameter validation functions."""
    
    def test_validate_positive_number_valid(self):
        """Test validate_positive_number with valid inputs."""
        assert validate_positive_number(1.0) == 1.0
        assert validate_positive_number(100) == 100
        assert validate_positive_number(0.001) == 0.001
    
    def test_validate_positive_number_invalid(self):
        """Test validate_positive_number with invalid inputs."""
        with pytest.raises(ValidationError, match="must be positive"):
            validate_positive_number(0)
        
        with pytest.raises(ValidationError, match="must be positive"):
            validate_positive_number(-1)
        
        with pytest.raises(ValidationError, match="must be positive"):
            validate_positive_number(-0.5)
    
    def test_validate_positive_number_with_name(self):
        """Test validate_positive_number with custom parameter name."""
        with pytest.raises(ValidationError, match="period must be positive"):
            validate_positive_number(-5, "period")
    
    def test_validate_non_negative_number_valid(self):
        """Test validate_non_negative_number with valid inputs."""
        assert validate_non_negative_number(0) == 0
        assert validate_non_negative_number(1.5) == 1.5
        assert validate_non_negative_number(100) == 100
    
    def test_validate_non_negative_number_invalid(self):
        """Test validate_non_negative_number with invalid inputs."""
        with pytest.raises(ValidationError, match="must be non-negative"):
            validate_non_negative_number(-1)
        
        with pytest.raises(ValidationError, match="must be non-negative"):
            validate_non_negative_number(-0.001)
    
    def test_validate_positive_integer_valid(self):
        """Test validate_positive_integer with valid inputs."""
        assert validate_positive_integer(1) == 1
        assert validate_positive_integer(100) == 100
    
    def test_validate_positive_integer_invalid(self):
        """Test validate_positive_integer with invalid inputs."""
        with pytest.raises(ValidationError, match="must be a positive integer"):
            validate_positive_integer(0)
        
        with pytest.raises(ValidationError, match="must be a positive integer"):
            validate_positive_integer(-1)
        
        with pytest.raises(ValidationError, match="must be a positive integer"):
            validate_positive_integer(1.5)
        
        with pytest.raises(ValidationError, match="must be a positive integer"):
            validate_positive_integer("1")
    
    def test_validate_range_valid(self):
        """Test validate_range with valid inputs."""
        assert validate_range(5, 1, 10) == 5
        assert validate_range(1, 1, 10) == 1  # Min boundary
        assert validate_range(10, 1, 10) == 10  # Max boundary
        assert validate_range(1.5, 1.0, 2.0) == 1.5
    
    def test_validate_range_invalid(self):
        """Test validate_range with invalid inputs."""
        with pytest.raises(ValidationError, match="must be between"):
            validate_range(0, 1, 10)
        
        with pytest.raises(ValidationError, match="must be between"):
            validate_range(11, 1, 10)
        
        with pytest.raises(ValidationError, match="must be between"):
            validate_range(-5, 0, 5)


class TestDataFrameValidation:
    """Test DataFrame validation functions."""
    
    def test_validate_dataframe_columns_valid(self):
        """Test validate_dataframe_columns with valid DataFrame."""
        df = pd.DataFrame({
            'open': [1.0, 1.1],
            'high': [1.2, 1.3],
            'low': [0.9, 1.0],
            'close': [1.1, 1.2],
            'volume': [1000, 1500]
        })
        
        required_columns = ['open', 'high', 'low', 'close']
        result = validate_dataframe_columns(df, required_columns)
        assert result == df
    
    def test_validate_dataframe_columns_missing(self):
        """Test validate_dataframe_columns with missing columns."""
        df = pd.DataFrame({
            'open': [1.0, 1.1],
            'high': [1.2, 1.3],
            'low': [0.9, 1.0]
            # Missing 'close' column
        })
        
        required_columns = ['open', 'high', 'low', 'close']
        with pytest.raises(ValidationError, match="Missing required columns"):
            validate_dataframe_columns(df, required_columns)
    
    def test_validate_dataframe_columns_empty_df(self):
        """Test validate_dataframe_columns with empty DataFrame."""
        df = pd.DataFrame()
        required_columns = ['open', 'high', 'low', 'close']
        
        with pytest.raises(ValidationError, match="DataFrame is empty"):
            validate_dataframe_columns(df, required_columns)
    
    def test_validate_series_numeric_valid(self):
        """Test validate_series_numeric with valid numeric series."""
        series = pd.Series([1.0, 2.5, 3.7, 4.2])
        result = validate_series_numeric(series)
        assert result.equals(series)
    
    def test_validate_series_numeric_with_nan(self):
        """Test validate_series_numeric with NaN values."""
        series = pd.Series([1.0, np.nan, 3.0, 4.0])
        
        # Should pass - NaN is acceptable by default
        result = validate_series_numeric(series, allow_nan=True)
        assert len(result) == 4
        
        # Should fail when NaN not allowed
        with pytest.raises(ValidationError, match="contains NaN values"):
            validate_series_numeric(series, allow_nan=False)
    
    def test_validate_series_numeric_non_numeric(self):
        """Test validate_series_numeric with non-numeric series."""
        series = pd.Series(['a', 'b', 'c'])
        
        with pytest.raises(ValidationError, match="not numeric"):
            validate_series_numeric(series)


class TestOHLCVValidation:
    """Test OHLCV data validation functions."""
    
    def create_valid_ohlcv(self):
        """Create a valid OHLCV DataFrame for testing."""
        return pd.DataFrame({
            'open': [1.0980, 1.0985, 1.0990],
            'high': [1.0995, 1.1000, 1.1005],
            'low': [1.0975, 1.0980, 1.0985],
            'close': [1.0985, 1.0990, 1.0995],
            'volume': [1000, 1500, 1200]
        })
    
    def test_validate_ohlcv_data_valid(self):
        """Test validate_ohlcv_data with valid data."""
        df = self.create_valid_ohlcv()
        result = validate_ohlcv_data(df)
        assert result.equals(df)
    
    def test_validate_ohlcv_data_missing_columns(self):
        """Test validate_ohlcv_data with missing required columns."""
        df = pd.DataFrame({
            'open': [1.0980, 1.0985],
            'high': [1.0995, 1.1000],
            'low': [1.0975, 1.0980]
            # Missing 'close' and 'volume'
        })
        
        with pytest.raises(ValidationError, match="Missing required OHLCV columns"):
            validate_ohlcv_data(df)
    
    def test_validate_ohlcv_data_negative_values(self):
        """Test validate_ohlcv_data with negative values."""
        df = pd.DataFrame({
            'open': [1.0980, -1.0985],  # Negative open price
            'high': [1.0995, 1.1000],
            'low': [1.0975, 1.0980],
            'close': [1.0985, 1.0990],
            'volume': [1000, 1500]
        })
        
        with pytest.raises(ValidationError, match="contains negative values"):
            validate_ohlcv_data(df)
    
    def test_validate_ohlcv_data_negative_volume(self):
        """Test validate_ohlcv_data with negative volume."""
        df = pd.DataFrame({
            'open': [1.0980, 1.0985],
            'high': [1.0995, 1.1000],
            'low': [1.0975, 1.0980],
            'close': [1.0985, 1.0990],
            'volume': [1000, -500]  # Negative volume
        })
        
        with pytest.raises(ValidationError, match="Volume cannot be negative"):
            validate_ohlcv_data(df)
    
    def test_validate_ohlcv_consistency_valid(self):
        """Test validate_ohlcv_consistency with valid data."""
        df = self.create_valid_ohlcv()
        result = validate_ohlcv_consistency(df)
        assert result.equals(df)
    
    def test_validate_ohlcv_consistency_high_low_invalid(self):
        """Test validate_ohlcv_consistency with high < low."""
        df = pd.DataFrame({
            'open': [1.0980, 1.0985],
            'high': [1.0970, 1.1000],  # High < Low for first row
            'low': [1.0975, 1.0980],
            'close': [1.0985, 1.0990],
            'volume': [1000, 1500]
        })
        
        with pytest.raises(ValidationError, match="High price is less than low price"):
            validate_ohlcv_consistency(df)
    
    def test_validate_ohlcv_consistency_price_outside_range(self):
        """Test validate_ohlcv_consistency with prices outside high-low range."""
        df = pd.DataFrame({
            'open': [1.1000, 1.0985],  # Open > High for first row
            'high': [1.0995, 1.1000],
            'low': [1.0975, 1.0980],
            'close': [1.0985, 1.0990],
            'volume': [1000, 1500]
        })
        
        with pytest.raises(ValidationError, match="Open price is outside high-low range"):
            validate_ohlcv_consistency(df)
    
    def test_validate_data_completeness_complete(self):
        """Test validate_data_completeness with complete data."""
        df = self.create_valid_ohlcv()
        result = validate_data_completeness(df)
        assert result == df
    
    def test_validate_data_completeness_with_gaps(self):
        """Test validate_data_completeness with gaps."""
        df = self.create_valid_ohlcv()
        df.loc[1, 'close'] = np.nan  # Introduce gap
        
        with pytest.raises(ValidationError, match="Data contains"):
            validate_data_completeness(df, max_gap_percentage=5.0)
    
    def test_validate_data_quality_good(self):
        """Test validate_data_quality with good quality data."""
        df = self.create_valid_ohlcv()
        result = validate_data_quality(df)
        assert result == df
    
    def test_validate_data_quality_poor(self):
        """Test validate_data_quality with poor quality data."""
        df = self.create_valid_ohlcv()
        # Introduce many NaN values
        df.loc[1:2, ['close', 'volume']] = np.nan
        
        with pytest.raises(ValidationError, match="Data quality score"):
            validate_data_quality(df, min_quality_score=0.8)


class TestIndicatorParameterValidation:
    """Test indicator parameter validation functions."""
    
    def test_validate_indicator_parameters_sma(self):
        """Test validating SMA parameters."""
        params = {'period': 20}
        result = validate_indicator_parameters(IndicatorType.SMA, params)
        assert result == params
    
    def test_validate_indicator_parameters_invalid_period(self):
        """Test validating parameters with invalid period."""
        params = {'period': -5}
        
        with pytest.raises(ValidationError, match="period must be positive"):
            validate_indicator_parameters(IndicatorType.SMA, params)
    
    def test_validate_indicator_parameters_rsi(self):
        """Test validating RSI parameters."""
        params = {'period': 14}
        result = validate_indicator_parameters(IndicatorType.RSI, params)
        assert result == params
    
    def test_validate_indicator_parameters_macd(self):
        """Test validating MACD parameters."""
        params = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        }
        result = validate_indicator_parameters(IndicatorType.MACD, params)
        assert result == params
    
    def test_validate_indicator_parameters_macd_invalid_periods(self):
        """Test validating MACD with invalid period relationship."""
        params = {
            'fast_period': 26,  # Fast > slow (invalid)
            'slow_period': 12,
            'signal_period': 9
        }
        
        with pytest.raises(ValidationError, match="Fast period must be less than slow period"):
            validate_indicator_parameters(IndicatorType.MACD, params)
    
    def test_validate_indicator_parameters_bollinger_bands(self):
        """Test validating Bollinger Bands parameters."""
        params = {
            'period': 20,
            'std_dev': 2.0
        }
        result = validate_indicator_parameters(IndicatorType.BBANDS, params)
        assert result == params
    
    def test_create_parameter_schema_sma(self):
        """Test creating parameter schema for SMA."""
        schema = create_parameter_schema(IndicatorType.SMA)
        
        assert 'period' in schema
        assert schema['period']['type'] == 'positive_integer'
        assert schema['period']['default'] == 20
    
    def test_create_parameter_schema_unknown_indicator(self):
        """Test creating schema for unknown indicator."""
        # Using a string instead of enum to test fallback
        schema = create_parameter_schema("unknown_indicator")
        assert schema == {}
    
    def test_validate_against_schema_valid(self):
        """Test validate_against_schema with valid parameters."""
        schema = {
            'period': {'type': 'positive_integer', 'default': 20},
            'multiplier': {'type': 'positive_number', 'default': 2.0}
        }
        params = {'period': 14, 'multiplier': 1.5}
        
        result = validate_against_schema(params, schema)
        assert result == params
    
    def test_validate_against_schema_with_defaults(self):
        """Test validate_against_schema applying defaults."""
        schema = {
            'period': {'type': 'positive_integer', 'default': 20},
            'multiplier': {'type': 'positive_number', 'default': 2.0}
        }
        params = {'period': 14}  # Missing multiplier
        
        result = validate_against_schema(params, schema)
        assert result['period'] == 14
        assert result['multiplier'] == 2.0
    
    def test_validate_against_schema_invalid_type(self):
        """Test validate_against_schema with invalid parameter type."""
        schema = {
            'period': {'type': 'positive_integer', 'default': 20}
        }
        params = {'period': 'invalid'}
        
        with pytest.raises(ValidationError):
            validate_against_schema(params, schema)


class TestTypeValidation:
    """Test type validation functions."""
    
    def test_validate_type_valid(self):
        """Test validate_type with valid type."""
        assert validate_type(5, int) == 5
        assert validate_type(1.5, float) == 1.5
        assert validate_type("test", str) == "test"
        assert validate_type([1, 2, 3], list) == [1, 2, 3]
    
    def test_validate_type_invalid(self):
        """Test validate_type with invalid type."""
        with pytest.raises(ValidationError, match="must be of type"):
            validate_type("5", int)
        
        with pytest.raises(ValidationError, match="must be of type"):
            validate_type(5, str)
    
    def test_validate_type_with_name(self):
        """Test validate_type with custom parameter name."""
        with pytest.raises(ValidationError, match="period must be of type"):
            validate_type("invalid", int, "period")
    
    def test_validate_enum_value_valid(self):
        """Test validate_enum_value with valid enum value."""
        result = validate_enum_value(Timeframe.H1, Timeframe)
        assert result == Timeframe.H1
        
        # Test with string value
        result = validate_enum_value("1h", Timeframe)
        assert result == Timeframe.H1
    
    def test_validate_enum_value_invalid(self):
        """Test validate_enum_value with invalid enum value."""
        with pytest.raises(ValidationError, match="must be one of"):
            validate_enum_value("invalid_timeframe", Timeframe)
        
        with pytest.raises(ValidationError, match="must be one of"):
            validate_enum_value(123, Timeframe)


class TestUtilityFunctions:
    """Test utility functions for data validation."""
    
    def test_check_data_gaps_no_gaps(self):
        """Test check_data_gaps with complete data."""
        df = pd.DataFrame({
            'close': [1.0, 1.1, 1.2, 1.3],
            'volume': [1000, 1100, 1200, 1300]
        })
        
        gaps = check_data_gaps(df)
        assert gaps['total_gaps'] == 0
        assert gaps['gap_percentage'] == 0.0
        assert gaps['columns_with_gaps'] == []
    
    def test_check_data_gaps_with_gaps(self):
        """Test check_data_gaps with missing data."""
        df = pd.DataFrame({
            'close': [1.0, np.nan, 1.2, np.nan],
            'volume': [1000, 1100, np.nan, 1300]
        })
        
        gaps = check_data_gaps(df)
        assert gaps['total_gaps'] == 3
        assert gaps['gap_percentage'] == 37.5  # 3 out of 8 total values
        assert 'close' in gaps['columns_with_gaps']
        assert 'volume' in gaps['columns_with_gaps']
    
    def test_calculate_data_quality_score_perfect(self):
        """Test calculate_data_quality_score with perfect data."""
        df = pd.DataFrame({
            'open': [1.0, 1.1, 1.2],
            'high': [1.1, 1.2, 1.3],
            'low': [0.9, 1.0, 1.1],
            'close': [1.05, 1.15, 1.25],
            'volume': [1000, 1100, 1200]
        })
        
        score = calculate_data_quality_score(df)
        assert score == 1.0  # Perfect score
    
    def test_calculate_data_quality_score_with_gaps(self):
        """Test calculate_data_quality_score with data gaps."""
        df = pd.DataFrame({
            'open': [1.0, np.nan, 1.2],
            'high': [1.1, 1.2, np.nan],
            'low': [0.9, 1.0, 1.1],
            'close': [1.05, 1.15, 1.25],
            'volume': [1000, 1100, 1200]
        })
        
        score = calculate_data_quality_score(df)
        assert score < 1.0  # Should be less than perfect
        assert score > 0.0  # Should be positive
    
    def test_calculate_data_quality_score_empty_data(self):
        """Test calculate_data_quality_score with empty DataFrame."""
        df = pd.DataFrame()
        
        score = calculate_data_quality_score(df)
        assert score == 0.0


class TestValidationIntegration:
    """Test integration between different validation functions."""
    
    def test_full_ohlcv_validation_pipeline(self):
        """Test complete OHLCV validation pipeline."""
        # Create valid data
        df = pd.DataFrame({
            'open': [1.0980, 1.0985, 1.0990],
            'high': [1.0995, 1.1000, 1.1005],
            'low': [1.0975, 1.0980, 1.0985],
            'close': [1.0985, 1.0990, 1.0995],
            'volume': [1000, 1500, 1200]
        })
        
        # Run through validation pipeline
        result = validate_ohlcv_data(df)
        result = validate_ohlcv_consistency(result)
        result = validate_data_completeness(result)
        result = validate_data_quality(result)
        
        assert result.equals(df)
    
    def test_indicator_validation_integration(self):
        """Test indicator parameter validation integration."""
        # Test multiple indicators
        indicators_and_params = [
            (IndicatorType.SMA, {'period': 20}),
            (IndicatorType.EMA, {'period': 12}),
            (IndicatorType.RSI, {'period': 14}),
            (IndicatorType.MACD, {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}),
            (IndicatorType.BBANDS, {'period': 20, 'std_dev': 2.0})
        ]
        
        for indicator_type, params in indicators_and_params:
            result = validate_indicator_parameters(indicator_type, params)
            assert isinstance(result, dict)
            assert all(key in result for key in params.keys())


if __name__ == "__main__":
    pytest.main([__file__]) 