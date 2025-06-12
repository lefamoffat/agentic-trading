"""
Indicator validation utilities for the agentic trading system.

This module provides validation functions for technical indicators,
their parameters, and related trading calculations.
"""

from typing import Dict, Any, Optional, Union

from src.types import IndicatorType
from src.exceptions import ValidationError, InvalidParameterError, format_validation_error, create_context
from .numeric import validate_positive_integer, validate_positive_number


def create_parameter_schema(indicator_type: Union[IndicatorType, str]) -> Dict[str, Any]:
    """
    Create parameter schema for an indicator type (alias for get_indicator_schema).
    
    Args:
        indicator_type: Type of indicator
        
    Returns:
        Dictionary containing parameter schema
    """
    # Handle string inputs (return empty schema for unknown indicators)
    if isinstance(indicator_type, str):
        return {}
    
    return get_indicator_schema(indicator_type)


def validate_against_schema(parameters: Dict[str, Any], 
                           schema: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate parameters against a schema (alias for _validate_params_against_schema).
    
    Args:
        parameters: Parameters to validate
        schema: Schema definition
        
    Returns:
        Validated parameters
        
    Raises:
        ValidationError: If validation fails
    """
    return _validate_params_against_schema(parameters, schema)


def get_indicator_schema(indicator_type: IndicatorType) -> Dict[str, Dict[str, Any]]:
    """
    Get the parameter schema for a specific indicator type.
    
    Args:
        indicator_type: Type of indicator
        
    Returns:
        Dictionary containing parameter schema
    """
    schemas = {
        IndicatorType.SMA: {
            'period': {'type': 'positive_integer', 'default': 20, 'min': 1, 'max': 1000}
        },
        IndicatorType.EMA: {
            'period': {'type': 'positive_integer', 'default': 12, 'min': 1, 'max': 1000}
        },
        IndicatorType.RSI: {
            'period': {'type': 'positive_integer', 'default': 14, 'min': 2, 'max': 100}
        },
        IndicatorType.MACD: {
            'fast_period': {'type': 'positive_integer', 'default': 12, 'min': 1, 'max': 100},
            'slow_period': {'type': 'positive_integer', 'default': 26, 'min': 1, 'max': 200},
            'signal_period': {'type': 'positive_integer', 'default': 9, 'min': 1, 'max': 50}
        },
        IndicatorType.BOLLINGER_BANDS: {
            'period': {'type': 'positive_integer', 'default': 20, 'min': 2, 'max': 100},
            'std_dev': {'type': 'positive_number', 'default': 2.0, 'min': 0.1, 'max': 5.0}
        },
        IndicatorType.STOCH: {
            'k_period': {'type': 'positive_integer', 'default': 14, 'min': 1, 'max': 100},
            'd_period': {'type': 'positive_integer', 'default': 3, 'min': 1, 'max': 100}
        }
    }
    
    return schemas.get(indicator_type, {})


def validate_indicator_parameters(indicator_type: IndicatorType, 
                                parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate parameters for a specific indicator.
    
    Args:
        indicator_type: Type of indicator
        parameters: Dictionary of parameters to validate
        
    Returns:
        Validated parameters dictionary
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(parameters, dict):
        raise ValidationError(f"Parameters must be a dictionary, got {type(parameters)}")
    
    # Get schema for this indicator type
    schema = get_indicator_schema(indicator_type)
    
    if not schema:
        # For unknown indicators, just validate that parameters is a dict
        return parameters
    
    # Custom validation for specific indicators
    if indicator_type == IndicatorType.SMA and 'period' in parameters:
        if parameters['period'] < 1:
            raise ValidationError("period must be positive")
    elif indicator_type == IndicatorType.MACD:
        fast = parameters.get('fast_period', 12)
        slow = parameters.get('slow_period', 26)
        if fast >= slow:
            raise ValidationError("Fast period must be less than slow period")
    
    return _validate_params_against_schema(parameters, schema)


def _validate_params_against_schema(parameters: Dict[str, Any], 
                                  schema: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate parameters against a schema definition.
    
    Args:
        parameters: Parameters to validate
        schema: Schema definition
        
    Returns:
        Validated parameters
        
    Raises:
        ValidationError: If validation fails
    """
    validated = {}
    
    # Type mapping for string type names
    type_mapping = {
        'positive_integer': int,
        'positive_number': (int, float),
        'integer': int,
        'number': (int, float),
        'string': str,
        'boolean': bool,
        'list': list,
        'dict': dict
    }
    
    for param_name, param_config in schema.items():
        if param_name in parameters:
            value = parameters[param_name]
            
            # Type validation
            expected_type_str = param_config.get('type')
            if expected_type_str:
                expected_type = type_mapping.get(expected_type_str, expected_type_str)
                
                if isinstance(expected_type, tuple):
                    # Multiple allowed types
                    if not isinstance(value, expected_type):
                        type_names = [t.__name__ for t in expected_type]
                        raise ValidationError(
                            f"Parameter '{param_name}' must be one of types {type_names}, got {type(value).__name__}"
                        )
                elif isinstance(expected_type, type):
                    # Single type
                    if not isinstance(value, expected_type):
                        raise ValidationError(
                            f"Parameter '{param_name}' must be of type {expected_type.__name__}, got {type(value).__name__}"
                        )
                # If it's still a string (unknown type), skip type validation
            
            # Range validation
            min_val = param_config.get('min')
            max_val = param_config.get('max')
            
            if min_val is not None and value < min_val:
                raise ValidationError(f"Parameter '{param_name}' must be >= {min_val}, got {value}")
            
            if max_val is not None and value > max_val:
                raise ValidationError(f"Parameter '{param_name}' must be <= {max_val}, got {value}")
            
            validated[param_name] = value
        elif param_config.get('required', False):
            raise ValidationError(f"Required parameter '{param_name}' is missing")
        elif 'default' in param_config:
            # Apply default value if parameter is missing
            validated[param_name] = param_config['default']
    
    # Add any extra parameters that aren't in the schema
    for param_name, value in parameters.items():
        if param_name not in schema:
            validated[param_name] = value
    
    return validated


def validate_moving_average_period(period: int) -> None:
    """
    Validate a moving average period parameter.
    
    Args:
        period: Period value to validate
        
    Raises:
        InvalidParameterError: If validation fails
    """
    try:
        validate_positive_integer(period, "period")
        
        if period > 1000:
            raise InvalidParameterError(
                format_validation_error("period", period, "integer <= 1000"),
                context=create_context(parameter="period", value=period, max_allowed=1000)
            )
    except ValidationError as e:
        raise InvalidParameterError(str(e), context=create_context(parameter="period", value=period))


def validate_rsi_period(period: int) -> None:
    """
    Validate an RSI period parameter.
    
    Args:
        period: Period value to validate
        
    Raises:
        InvalidParameterError: If validation fails
    """
    try:
        validate_positive_integer(period, "period")
        
        if period < 2:
            raise InvalidParameterError(
                format_validation_error("period", period, "integer >= 2"),
                context=create_context(parameter="period", value=period, min_allowed=2)
            )
        
        if period > 100:
            raise InvalidParameterError(
                format_validation_error("period", period, "integer <= 100"),
                context=create_context(parameter="period", value=period, max_allowed=100)
            )
    except ValidationError as e:
        raise InvalidParameterError(str(e), context=create_context(parameter="period", value=period))


def validate_bollinger_bands_parameters(period: int, std_dev: float) -> None:
    """
    Validate Bollinger Bands parameters.
    
    Args:
        period: Period for moving average
        std_dev: Standard deviation multiplier
        
    Raises:
        InvalidParameterError: If validation fails
    """
    try:
        validate_moving_average_period(period)
        validate_positive_number(std_dev, "std_dev")
        
        if std_dev > 5.0:
            raise InvalidParameterError(
                format_validation_error("std_dev", std_dev, "number <= 5.0"),
                context=create_context(parameter="std_dev", value=std_dev, max_allowed=5.0)
            )
    except ValidationError as e:
        raise InvalidParameterError(str(e))


def validate_macd_parameters(fast_period: int, slow_period: int, signal_period: int) -> None:
    """
    Validate MACD parameters.
    
    Args:
        fast_period: Fast EMA period
        slow_period: Slow EMA period  
        signal_period: Signal line EMA period
        
    Raises:
        InvalidParameterError: If validation fails
    """
    try:
        validate_positive_integer(fast_period, "fast_period")
        validate_positive_integer(slow_period, "slow_period")
        validate_positive_integer(signal_period, "signal_period")
        
        if fast_period >= slow_period:
            raise InvalidParameterError(
                "Fast period must be less than slow period",
                context=create_context(fast_period=fast_period, slow_period=slow_period)
            )
    except ValidationError as e:
        raise InvalidParameterError(str(e)) 