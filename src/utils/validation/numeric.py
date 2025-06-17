"""Numeric validation utilities for the agentic trading system.

This module provides validation functions for numeric values including
positive numbers, integers, ranges, and percentages.
"""

from typing import Optional, Union

from src.utils.exceptions import (
    InvalidParameterError,
    ValidationError,
    create_context,
    format_validation_error,
)


def validate_positive_number(
    value: Union[int, float], name: str = "value", allow_zero: bool = False
) -> Union[int, float]:
    """Validate that a value is a positive number.

    Args:
        value: Value to validate.
        name: Parameter name for error messages.
        allow_zero: Whether to allow zero values.

    Returns:
        The validated value.

    Raises:
        ValidationError: If validation fails.

    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(value).__name__}")

    if allow_zero:
        if value < 0:
            raise ValidationError(f"{name} must be non-negative, got {value}")
    else:
        if value <= 0:
            raise ValidationError(f"{name} must be positive, got {value}")

    return value


def validate_non_negative_number(
    value: Union[int, float], name: str = "value"
) -> Union[int, float]:
    """Validate that a value is a non-negative number (>= 0).

    Args:
        value: Value to validate.
        name: Parameter name for error messages.

    Returns:
        The validated value.

    Raises:
        ValidationError: If validation fails.

    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(value).__name__}")
    if value < 0:
        raise ValidationError(f"{name} must be non-negative, got {value}")
    return value


def validate_positive_integer(value: int, name: str = "value") -> int:
    """Validate that a value is a positive integer.

    Args:
        value: Value to validate.
        name: Parameter name for error messages.

    Returns:
        The validated value.

    Raises:
        ValidationError: If validation fails.

    """
    if not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValidationError(f"{name} must be a positive integer, got {value}")
    return value


def validate_range(
    value: Union[int, float],
    min_val: Union[int, float],
    max_val: Union[int, float],
    name: str = "value",
) -> Union[int, float]:
    """Validate that a value is within a specified range.

    Args:
        value: Value to validate.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).
        name: Parameter name for error messages.

    Returns:
        The validated value.

    Raises:
        ValidationError: If validation fails.

    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(value).__name__}")

    if value < min_val or value > max_val:
        raise ValidationError(
            f"{name} must be between {min_val} and {max_val}, got {value}"
        )

    return value


def validate_integer_range(
    value: int, name: str, min_val: Optional[int] = None, max_val: Optional[int] = None
) -> None:
    """Validate that an integer is within a specified range.

    Args:
        value: Value to validate.
        name: Parameter name for error messages.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive), None for no limit.

    Raises:
        InvalidParameterError: If validation fails.

    """
    if not isinstance(value, int):
        raise InvalidParameterError(
            format_validation_error(name, value, "an integer"),
            context=create_context(parameter=name, value=value, type=type(value)),
        )
    if min_val is not None and value < min_val:
        raise InvalidParameterError(
            format_validation_error(name, value, f"at least {min_val}"),
            context=create_context(parameter=name, value=value, min_val=min_val)
        )
    if max_val is not None and value > max_val:
        raise InvalidParameterError(
            format_validation_error(name, value, f"at most {max_val}"),
            context=create_context(parameter=name, value=value, max_val=max_val)
        )


def validate_percentage(
    value: float, name: str, allow_negative: bool = False
) -> None:
    """Validate that a value is a valid percentage (0-100 or -100 to 100).

    Args:
        value: Value to validate.
        name: Parameter name for error messages.
        allow_negative: Whether to allow negative percentages.

    Raises:
        InvalidParameterError: If validation fails.

    """
    if not isinstance(value, (int, float)):
        raise InvalidParameterError(
            format_validation_error(name, value, "a number"),
            context=create_context(parameter=name, value=value, type=type(value))
        )
    min_val, max_val = (-100, 100) if allow_negative else (0, 100)
    if not min_val <= value <= max_val:
        raise InvalidParameterError(
            format_validation_error(name, value, f"between {min_val}% and {max_val}%"),
            context=create_context(
                parameter=name, value=value, min_val=min_val, max_val=max_val
            )
        )
