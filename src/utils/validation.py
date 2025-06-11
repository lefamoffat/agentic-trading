"""
Validation utilities for the agentic trading system.

This module provides common validation functions used across different
components for parameter validation, data validation, and input sanitization.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple, Type
from datetime import datetime

# Use absolute imports
from types import ValidationResult, OHLCVData, SymbolType, BrokerType
from exceptions import (
    ValidationError, DataFormatError, InvalidParameterError,
    MissingDataError, format_validation_error, create_context
)


def validate_positive_number(value: Union[int, float], name: str, 
                           allow_zero: bool = False) -> None:
    """
    Validate that a value is a positive number.
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        allow_zero: Whether to allow zero values
        
    Raises:
        InvalidParameterError: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise InvalidParameterError(
            format_validation_error(name, value, "numeric type"),
            context=create_context(parameter=name, value=value)
        )
    
    if np.isnan(value) or np.isinf(value):
        raise InvalidParameterError(
            format_validation_error(name, value, "finite number"),
            context=create_context(parameter=name, value=value)
        )
    
    min_value = 0 if allow_zero else 0.000001
    if value < min_value:
        expected = "positive number" + (" or zero" if allow_zero else "")
        raise InvalidParameterError(
            format_validation_error(name, value, expected),
            context=create_context(parameter=name, value=value, min_value=min_value)
        )


def validate_integer_range(value: int, name: str, min_val: int, 
                          max_val: Optional[int] = None) -> None:
    """
    Validate that an integer is within a specified range.
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive), None for no limit
        
    Raises:
        InvalidParameterError: If validation fails
    """
    if not isinstance(value, int):
        raise InvalidParameterError(
            format_validation_error(name, value, "integer"),
            context=create_context(parameter=name, value=value)
        )
    
    if value < min_val:
        raise InvalidParameterError(
            format_validation_error(name, value, f"integer >= {min_val}"),
            context=create_context(parameter=name, value=value, min_val=min_val)
        )
    
    if max_val is not None and value > max_val:
        raise InvalidParameterError(
            format_validation_error(name, value, f"integer <= {max_val}"),
            context=create_context(parameter=name, value=value, max_val=max_val)
        )


def validate_percentage(value: float, name: str, allow_negative: bool = False) -> None:
    """
    Validate that a value is a valid percentage (0-100 or -100 to 100).
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        allow_negative: Whether to allow negative percentages
        
    Raises:
        InvalidParameterError: If validation fails
    """
    validate_positive_number(value, name, allow_zero=True)
    
    min_val = -100.0 if allow_negative else 0.0
    max_val = 100.0
    
    if value < min_val or value > max_val:
        range_str = f"{min_val}-{max_val}"
        raise InvalidParameterError(
            format_validation_error(name, value, f"percentage in range {range_str}"),
            context=create_context(parameter=name, value=value, range=range_str)
        )


def validate_string_enum(value: str, name: str, valid_values: List[str], 
                        case_sensitive: bool = True) -> None:
    """
    Validate that a string is one of the allowed values.
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        valid_values: List of valid string values
        case_sensitive: Whether comparison is case sensitive
        
    Raises:
        InvalidParameterError: If validation fails
    """
    if not isinstance(value, str):
        raise InvalidParameterError(
            format_validation_error(name, value, "string"),
            context=create_context(parameter=name, value=value)
        )
    
    comparison_values = valid_values if case_sensitive else [v.lower() for v in valid_values]
    comparison_value = value if case_sensitive else value.lower()
    
    if comparison_value not in comparison_values:
        raise InvalidParameterError(
            format_validation_error(name, value, f"one of {valid_values}"),
            context=create_context(parameter=name, value=value, valid_values=valid_values)
        )


def validate_symbol(symbol: str) -> None:
    """
    Validate that a symbol string is properly formatted.
    
    Args:
        symbol: Trading symbol to validate
        
    Raises:
        InvalidParameterError: If symbol format is invalid
    """
    if not isinstance(symbol, str):
        raise InvalidParameterError(
            format_validation_error("symbol", symbol, "string"),
            context=create_context(symbol=symbol)
        )
    
    if not symbol or not symbol.strip():
        raise InvalidParameterError(
            "Symbol cannot be empty",
            context=create_context(symbol=symbol)
        )
    
    # Remove common separators and check remaining characters
    cleaned = symbol.replace("/", "").replace("-", "").replace("_", "")
    if not cleaned.isalnum():
        raise InvalidParameterError(
            "Symbol must contain only alphanumeric characters and common separators (/, -, _)",
            context=create_context(symbol=symbol)
        )


def validate_ohlcv_data(data: pd.DataFrame, required_columns: Optional[List[str]] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate OHLCV data format and completeness.
    
    Args:
        data: DataFrame to validate
        required_columns: List of required columns, defaults to OHLCV
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if required_columns is None:
        required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    try:
        # Check if input is DataFrame
        if not isinstance(data, pd.DataFrame):
            return False, f"Expected pandas DataFrame, got {type(data)}"
        
        # Check if DataFrame is empty
        if data.empty:
            return False, "DataFrame is empty"
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
        
        # Check for numeric data types
        numeric_columns = [col for col in required_columns if col != 'symbol']
        for col in numeric_columns:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                return False, f"Column {col} must be numeric"
        
        # Check for valid OHLC relationships (if all present)
        ohlc_cols = ['open', 'high', 'low', 'close']
        if all(col in data.columns for col in ohlc_cols):
            # High should be >= Open, Close, Low
            if not (data['high'] >= data[['open', 'close', 'low']].max(axis=1)).all():
                return False, "High prices must be >= Open, Close, and Low prices"
            
            # Low should be <= Open, Close, High
            if not (data['low'] <= data[['open', 'close', 'high']].min(axis=1)).all():
                return False, "Low prices must be <= Open, Close, and High prices"
        
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns and (data[col] <= 0).any():
                return False, f"Column {col} contains non-positive values"
        
        # Check for negative volume
        if 'volume' in data.columns and (data['volume'] < 0).any():
            return False, "Volume cannot be negative"
        
        # Check for excessive NaN values
        nan_percentage = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if nan_percentage > 0.1:  # More than 10% NaN
            return False, f"Excessive missing data: {nan_percentage:.1%} of values are NaN"
        
        return True, None
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_data_completeness(data: pd.DataFrame, min_rows: int = 1,
                             max_nan_percentage: float = 0.05) -> None:
    """
    Validate data completeness requirements.
    
    Args:
        data: DataFrame to validate
        min_rows: Minimum number of rows required
        max_nan_percentage: Maximum allowed percentage of NaN values
        
    Raises:
        MissingDataError: If data completeness requirements not met
    """
    if len(data) < min_rows:
        raise MissingDataError(
            f"Insufficient data: {len(data)} rows, minimum {min_rows} required",
            context=create_context(rows=len(data), min_rows=min_rows)
        )
    
    nan_count = data.isnull().sum().sum()
    total_values = len(data) * len(data.columns)
    nan_percentage = nan_count / total_values if total_values > 0 else 0
    
    if nan_percentage > max_nan_percentage:
        raise MissingDataError(
            f"Too much missing data: {nan_percentage:.1%}, maximum {max_nan_percentage:.1%} allowed",
            context=create_context(
                nan_percentage=nan_percentage,
                max_allowed=max_nan_percentage,
                nan_count=nan_count,
                total_values=total_values
            )
        )


def validate_indicator_parameters(parameters: Dict[str, Any], 
                                schema: Dict[str, Dict[str, Any]]) -> None:
    """
    Validate indicator parameters against a schema.
    
    Args:
        parameters: Parameter dictionary to validate
        schema: Validation schema with parameter definitions
        
    Schema format:
        {
            'parameter_name': {
                'type': int|float|str|bool,
                'min': minimum_value (optional),
                'max': maximum_value (optional),
                'valid_values': list_of_valid_values (optional),
                'required': bool (default True)
            }
        }
        
    Raises:
        InvalidParameterError: If validation fails
    """
    for param_name, param_schema in schema.items():
        is_required = param_schema.get('required', True)
        param_type = param_schema['type']
        
        # Check if required parameter is missing
        if is_required and param_name not in parameters:
            raise InvalidParameterError(
                f"Required parameter '{param_name}' is missing",
                context=create_context(parameter=param_name, schema=param_schema)
            )
        
        # Skip validation if parameter is not provided and not required
        if param_name not in parameters:
            continue
        
        value = parameters[param_name]
        
        # Type validation
        if not isinstance(value, param_type):
            raise InvalidParameterError(
                format_validation_error(param_name, value, param_type.__name__),
                context=create_context(parameter=param_name, value=value, expected_type=param_type)
            )
        
        # Range validation for numeric types
        if param_type in (int, float):
            if 'min' in param_schema and value < param_schema['min']:
                raise InvalidParameterError(
                    format_validation_error(param_name, value, f">= {param_schema['min']}"),
                    context=create_context(parameter=param_name, value=value, min_value=param_schema['min'])
                )
            
            if 'max' in param_schema and value > param_schema['max']:
                raise InvalidParameterError(
                    format_validation_error(param_name, value, f"<= {param_schema['max']}"),
                    context=create_context(parameter=param_name, value=value, max_value=param_schema['max'])
                )
        
        # Valid values validation
        if 'valid_values' in param_schema:
            valid_values = param_schema['valid_values']
            if value not in valid_values:
                raise InvalidParameterError(
                    format_validation_error(param_name, value, f"one of {valid_values}"),
                    context=create_context(parameter=param_name, value=value, valid_values=valid_values)
                )


def sanitize_string(value: str, max_length: int = 100, 
                   allowed_chars: Optional[str] = None) -> str:
    """
    Sanitize string input for safe usage.
    
    Args:
        value: String to sanitize
        max_length: Maximum allowed length
        allowed_chars: String of allowed characters, None for alphanumeric + common symbols
        
    Returns:
        Sanitized string
        
    Raises:
        ValidationError: If string cannot be sanitized
    """
    if not isinstance(value, str):
        raise ValidationError(f"Expected string, got {type(value)}")
    
    # Default allowed characters (alphanumeric + common trading symbols)
    if allowed_chars is None:
        allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_/.:"
    
    # Remove disallowed characters
    sanitized = ''.join(char for char in value if char in allowed_chars)
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    # Check if anything remains
    if not sanitized:
        raise ValidationError(f"String '{value}' contains no valid characters")
    
    return sanitized


def validate_timeframe_compatibility(timeframes: List[str]) -> None:
    """
    Validate that multiple timeframes are compatible for analysis.
    
    Args:
        timeframes: List of timeframe strings
        
    Raises:
        InvalidParameterError: If timeframes are incompatible
    """
    if not timeframes:
        raise InvalidParameterError("At least one timeframe must be specified")
    
    # Define timeframe hierarchy (in minutes)
    timeframe_minutes = {
        '1m': 1, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '4h': 240, '1d': 1440, '1w': 10080, '1M': 43200
    }
    
    invalid_timeframes = [tf for tf in timeframes if tf not in timeframe_minutes]
    if invalid_timeframes:
        raise InvalidParameterError(
            f"Invalid timeframes: {invalid_timeframes}",
            context=create_context(
                invalid_timeframes=invalid_timeframes,
                valid_timeframes=list(timeframe_minutes.keys())
            )
        ) 