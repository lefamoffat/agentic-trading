"""
Numeric validation utilities for the agentic trading system.

This module provides validation functions for numeric values including
positive numbers, integers, ranges, and percentages.
"""

import numpy as np
from typing import Union, Optional

from src.exceptions import ValidationError, InvalidParameterError, format_validation_error, create_context


def validate_positive_number(value: Union[int, float], name: str = "value", 
                           allow_zero: bool = False) -> Union[int, float]:
    """
    Validate that a value is a positive number.
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        allow_zero: Whether to allow zero values
        
    Returns:
        The validated value
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(value)}")
    
    if np.isnan(value) or np.isinf(value):
        raise ValidationError(f"{name} must be a finite number")
    
    if allow_zero and value < 0:
        raise ValidationError(f"{name} must be non-negative")
    elif not allow_zero and value <= 0:
        raise ValidationError(f"{name} must be positive")
    
    return value


def validate_non_negative_number(value: Union[int, float], name: str = "value") -> Union[int, float]:
    """
    Validate that a value is a non-negative number (>= 0).
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        
    Returns:
        The validated value
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(value)}")
    
    if np.isnan(value) or np.isinf(value):
        raise ValidationError(f"{name} must be a finite number")
    
    if value < 0:
        raise ValidationError(f"{name} must be non-negative")
    
    return value


def validate_positive_integer(value: int, name: str = "value") -> int:
    """
    Validate that a value is a positive integer.
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        
    Returns:
        The validated value
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, int):
        raise ValidationError(f"{name} must be a positive integer, got {type(value)}")
    
    if value <= 0:
        raise ValidationError(f"{name} must be a positive integer")
    
    return value


def validate_range(value: Union[int, float], min_val: Union[int, float], 
                   max_val: Union[int, float], name: str = "value") -> Union[int, float]:
    """
    Validate that a value is within a specified range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        name: Parameter name for error messages
        
    Returns:
        The validated value
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(value)}")
    
    if value < min_val or value > max_val:
        raise ValidationError(f"{name} must be between {min_val} and {max_val}, got {value}")
    
    return value


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