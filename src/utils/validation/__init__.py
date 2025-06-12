"""
Validation utilities for the agentic trading system.

This module provides a comprehensive set of validation functions for different
aspects of the trading system including numeric values, DataFrames, types,
and trading-specific validations.

The validation functions are organized into focused sub-modules:
- numeric: Numeric value validation
- dataframe: DataFrame and OHLCV data validation  
- types: Type and enum validation
- indicators: Technical indicator validation
"""

# Re-export all validation functions for backward compatibility
from .numeric import (
    validate_positive_number,
    validate_non_negative_number, 
    validate_positive_integer,
    validate_range,
    validate_integer_range,
    validate_percentage
)

from .dataframe import (
    validate_dataframe_columns,
    validate_series_numeric,
    validate_ohlcv_data,
    validate_ohlcv_consistency,
    validate_data_completeness,
    validate_data_quality,
    calculate_data_quality_score,
    check_data_gaps
)

from .types import (
    validate_type,
    validate_enum_value,
    validate_string_enum,
    validate_symbol,
    sanitize_string
)

from .indicators import (
    get_indicator_schema,
    create_parameter_schema,
    validate_indicator_parameters,
    validate_against_schema,
    validate_moving_average_period,
    validate_rsi_period,
    validate_bollinger_bands_parameters,
    validate_macd_parameters
)

# List of all exported functions for convenience
__all__ = [
    # Numeric validation
    'validate_positive_number',
    'validate_non_negative_number',
    'validate_positive_integer', 
    'validate_range',
    'validate_integer_range',
    'validate_percentage',
    
    # DataFrame validation
    'validate_dataframe_columns',
    'validate_series_numeric',
    'validate_ohlcv_data',
    'validate_ohlcv_consistency',
    'validate_data_completeness',
    'validate_data_quality',
    'calculate_data_quality_score',
    'check_data_gaps',
    
    # Type validation
    'validate_type',
    'validate_enum_value',
    'validate_string_enum',
    'validate_symbol',
    'sanitize_string',
    
    # Indicator validation
    'get_indicator_schema',
    'create_parameter_schema',
    'validate_indicator_parameters',
    'validate_against_schema',
    'validate_moving_average_period',
    'validate_rsi_period',
    'validate_bollinger_bands_parameters',
    'validate_macd_parameters'
] 