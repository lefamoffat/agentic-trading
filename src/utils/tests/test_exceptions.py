"""Tests for the project-wide exceptions module (relocated)."""

import unittest

import pytest

from src.utils.exceptions import (
    BrokerAPIError,
    ConfigurationError,
    DataError,
    FeatureEngineeringError,
    IndicatorCalculationError,
    InvalidParameterError,
    MissingConfigError,
    MissingDataError,
    OrderExecutionError,
    SignalGenerationError,
    SlippageError,
    StrategyError,
    TradingExecutionError,
    TradingSystemError,
    ValidationError,
)

@pytest.mark.unit
class TestExceptionSystem(unittest.TestCase):
    def test_base_exception_inheritance(self):
        for exc in [ConfigurationError, DataError, StrategyError, BrokerAPIError, ValidationError]:
            assert issubclass(exc, TradingSystemError)

    def test_specific_inheritance(self):
        assert issubclass(MissingConfigError, ConfigurationError)
        assert issubclass(InvalidParameterError, FeatureEngineeringError)
        assert issubclass(MissingDataError, DataError)
        assert issubclass(IndicatorCalculationError, FeatureEngineeringError)
        assert issubclass(SignalGenerationError, StrategyError)
        assert issubclass(OrderExecutionError, TradingExecutionError)
        assert issubclass(SlippageError, TradingExecutionError)

    def test_context_storage(self):
        ctx = {"foo": "bar"}
        err = TradingSystemError("oops", context=ctx)
        assert err.context == ctx
