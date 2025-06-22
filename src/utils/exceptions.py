# -*- coding: utf-8 -*-
"""Lean project-wide exception hierarchy.

Only the exception classes actually referenced in code/tests are defined to
avoid dead weight while still providing semantic granularity for callers.
"""
from typing import Any, Dict, Optional

class TradingSystemError(Exception):
    """Root of all custom exceptions in the trading system."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context: Dict[str, Any] = context or {}

    def __str__(self) -> str:  # noqa: DunderStr
        base = super().__str__()
        return f"{base} | context={self.context}" if self.context else base

# ────────────────────────
# Data layer
# ────────────────────────
class DataError(TradingSystemError):
    """Generic data-related error."""

class ValidationError(DataError):
    """Input failed validation rules."""

class MissingDataError(DataError):
    """Required data is absent."""

# ────────────────────────
# Feature engineering
# ────────────────────────
class FeatureEngineeringError(TradingSystemError):
    pass

class InvalidParameterError(FeatureEngineeringError):
    pass

class IndicatorCalculationError(FeatureEngineeringError):
    pass

# ────────────────────────
# Configuration
# ────────────────────────
class ConfigurationError(TradingSystemError):
    pass

class MissingConfigError(ConfigurationError):
    pass

# ────────────────────────
# Broker layer
# ────────────────────────
class BrokerError(TradingSystemError):
    pass

class BrokerConnectionError(BrokerError):
    pass

class BrokerAuthenticationError(BrokerError):
    pass

class BrokerAPIError(BrokerError):
    def __init__(self, message: str, api_code: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        ctx = {"api_code": api_code, **(context or {})}
        super().__init__(message, ctx)
        self.api_code = api_code

# ────────────────────────
# Strategy layer
# ────────────────────────
class StrategyError(TradingSystemError):
    pass

class SignalGenerationError(StrategyError):
    pass

# ────────────────────────
# Execution layer
# ────────────────────────
class TradingExecutionError(TradingSystemError):
    pass

class OrderExecutionError(TradingExecutionError):
    pass

class SlippageError(TradingExecutionError):
    pass

# Convenience export list
__all__ = [
    "BrokerAPIError",
    "BrokerAuthenticationError",
    "BrokerConnectionError",
    "BrokerError",
    "ConfigurationError",
    "DataError",
    "FeatureEngineeringError",
    "IndicatorCalculationError",
    "InvalidParameterError",
    "MissingConfigError",
    "MissingDataError",
    "OrderExecutionError",
    "SignalGenerationError",
    "SlippageError",
    "StrategyError",
    "TradingExecutionError",
    "TradingSystemError",
    "ValidationError",
    "create_context",
    "format_validation_error",
]

# ---------------------------------------------------------------------------
# Utility helpers (still centralised here to avoid tiny import modules)
# ---------------------------------------------------------------------------

def create_context(**kwargs) -> Dict[str, Any]:
    """Return kwargs unmodified - syntactic sugar for populating exception context."""
    return kwargs

def format_validation_error(field_name: str, value: Any, expected: str) -> str:
    """Small helper to produce consistent validation-error messages."""
    return f"Invalid {field_name}: got {value}, expected {expected}"
