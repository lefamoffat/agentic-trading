"""Type validation utilities for the agentic trading system.

This module provides validation functions for types, enums, strings,
and other data type validation scenarios.
"""

import re
from typing import Any, List, Optional, Type

from src.utils.exceptions import (
    InvalidParameterError,
    ValidationError,
    create_context,
    format_validation_error,
)

def validate_type(value: Any, expected_type: Type, name: str = "value") -> Any:
    """Validate that a value is of the expected type.

    Args:
        value: Value to validate.
        expected_type: Expected type.
        name: Parameter name for error messages.

    Returns:
        The validated value.

    Raises:
        ValidationError: If validation fails.

    """
    if not isinstance(value, expected_type):
        raise ValidationError(
            f"{name} must be of type {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )

    return value

def validate_enum_value(value: Any, enum_class: Type, name: str = "value") -> Any:
    """Validate that a value is a valid enum member or can be converted to one.

    Args:
        value: Value to validate (can be enum member or string).
        enum_class: Enum class to validate against.
        name: Parameter name for error messages.

    Returns:
        Valid enum member.

    Raises:
        ValidationError: If validation fails.

    """
    if isinstance(value, enum_class):
        return value

    if isinstance(value, str):
        try:
            return enum_class[value.upper()]
        except KeyError as err:
            valid_values = [e.name for e in enum_class]
            raise ValidationError(
                f"Invalid {name}: '{value}'. Must be one of {valid_values}",
                context=create_context(
                    parameter=name, value=value, valid_values=valid_values
                ),
            ) from err

    raise ValidationError(
        f"{name} must be a {enum_class.__name__} or a corresponding string, "
        f"got {type(value).__name__}"
    )

def validate_string_choice(
    value: str, name: str, valid_values: List[str], case_sensitive: bool = True
) -> None:
    """Validate that a string is one of the allowed values.

    Args:
        value: Value to validate.
        name: Parameter name for error messages.
        valid_values: List of valid string values.
        case_sensitive: Whether comparison is case sensitive.

    Raises:
        InvalidParameterError: If validation fails.

    """
    if not isinstance(value, str):
        raise InvalidParameterError(
            format_validation_error(name, value, "a string"),
            context=create_context(parameter=name, value=value),
        )

    comparison_values = (
        valid_values if case_sensitive else [v.lower() for v in valid_values]
    )
    comparison_value = value if case_sensitive else value.lower()

    if comparison_value not in comparison_values:
        raise InvalidParameterError(
            format_validation_error(name, value, f"one of {valid_values}"),
            context=create_context(
                parameter=name, value=value, valid_values=valid_values
            ),
        )

def validate_symbol(symbol: str) -> None:
    """Validate symbol format.

    Args:
        symbol: Symbol string to validate.

    Raises:
        ValidationError: If symbol format is invalid.

    """
    if not isinstance(symbol, str) or not re.match(r"^[A-Z0-9/]{3,12}$", symbol):
        raise ValidationError(
            f"Invalid symbol format: '{symbol}'. "
            "Expected 3-12 uppercase alphanumeric characters, e.g., 'EUR/USD'."
        )

def sanitize_string(
    value: str,
    name: str,
    max_length: int = 256,
    allowed_chars: Optional[str] = None,
) -> str:
    """Sanitize a string value for safe usage.

    Args:
        value: String to sanitize.
        name: The name of the parameter being sanitized.
        max_length: Maximum allowed length.
        allowed_chars: String of allowed characters (None for alphanumeric + basic
            punctuation).

    Returns:
        Sanitized string.

    Raises:
        ValidationError: If sanitization fails.

    """
    if not isinstance(value, str):
        raise ValidationError(f"Value for {name} must be a string.")

    # Truncate to max_length
    sanitized = value[:max_length]

    # Character validation
    if allowed_chars:
        allowed = set(allowed_chars)
        sanitized = "".join(c for c in sanitized if c in allowed)
    else:
        # Default: alphanumeric + basic punctuation
        allowed = set(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789 .,;:!?-_/"
        )
        sanitized = "".join(c for c in sanitized if c in allowed)

    return sanitized
