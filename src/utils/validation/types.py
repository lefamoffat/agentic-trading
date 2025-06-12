"""
Type validation utilities for the agentic trading system.

This module provides validation functions for types, enums, strings,
and other data type validation scenarios.
"""

from typing import Any, Type, List, Optional

from src.exceptions import ValidationError, InvalidParameterError, format_validation_error, create_context


def validate_type(value: Any, expected_type: Type, name: str = "value") -> Any:
    """
    Validate that a value is of the expected type.
    
    Args:
        value: Value to validate
        expected_type: Expected type
        name: Parameter name for error messages
        
    Returns:
        The validated value
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, expected_type):
        raise ValidationError(f"{name} must be of type {expected_type.__name__}, got {type(value).__name__}")
    
    return value


def validate_enum_value(value: Any, enum_class: Type, name: str = "value") -> Any:
    """
    Validate that a value is a valid enum member or can be converted to one.
    
    Args:
        value: Value to validate (can be enum member or string)
        enum_class: Enum class to validate against
        name: Parameter name for error messages
        
    Returns:
        Valid enum member
        
    Raises:
        ValidationError: If validation fails
    """
    # If already the correct enum type, return it
    if isinstance(value, enum_class):
        return value
    
    # Try to convert string to enum
    if isinstance(value, str):
        # Try by value first
        for member in enum_class:
            if member.value == value:
                return member
        
        # Try by name (case insensitive)
        value_upper = value.upper()
        for member in enum_class:
            if member.name.upper() == value_upper:
                return member
    
    # If we get here, validation failed
    valid_values = [member.value for member in enum_class]
    raise ValidationError(
        f"{name} must be one of {valid_values}, got {value}"
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
    Validate symbol format.
    
    Args:
        symbol: Symbol string to validate
        
    Raises:
        ValidationError: If symbol format is invalid
    """
    if not isinstance(symbol, str):
        raise ValidationError(f"Symbol must be a string, got {type(symbol)}")
    
    if not symbol:
        raise ValidationError("Symbol cannot be empty")
    
    # Basic validation - could be extended with more specific rules
    if len(symbol) < 3:
        raise ValidationError(f"Symbol '{symbol}' too short (minimum 3 characters)")
    
    if len(symbol) > 12:
        raise ValidationError(f"Symbol '{symbol}' too long (maximum 12 characters)")


def sanitize_string(value: str, max_length: int = 100, 
                   allowed_chars: Optional[str] = None) -> str:
    """
    Sanitize a string value for safe usage.
    
    Args:
        value: String to sanitize
        max_length: Maximum allowed length
        allowed_chars: String of allowed characters (None for alphanumeric + basic punctuation)
        
    Returns:
        Sanitized string
        
    Raises:
        ValidationError: If sanitization fails
    """
    if not isinstance(value, str):
        raise ValidationError(f"Value must be a string, got {type(value)}")
    
    # Trim whitespace
    sanitized = value.strip()
    
    # Check length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    # Filter allowed characters
    if allowed_chars is not None:
        sanitized = ''.join(c for c in sanitized if c in allowed_chars)
    else:
        # Default: alphanumeric + basic punctuation
        allowed = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?-_/')
        sanitized = ''.join(c for c in sanitized if c in allowed)
    
    return sanitized 