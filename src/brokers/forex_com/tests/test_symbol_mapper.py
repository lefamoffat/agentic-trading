"""Forex.com-specific tests for SymbolMapper (moved from generic tests)."""

import pytest
from src.brokers.symbol_mapper import BrokerType, SymbolMapper


@pytest.mark.unit
class TestForexComSymbolMapper:
    def setup_class(self):
        self.mapper = SymbolMapper(BrokerType.FOREX_COM)

    def test_major_pairs(self):
        pairs = [
            ("EUR/USD", "EUR_USD"),
            ("GBP/USD", "GBP_USD"),
            ("USD/JPY", "USD_JPY"),
        ]
        for common, broker in pairs:
            assert self.mapper.to_broker_symbol(common) == broker
            assert self.mapper.from_broker_symbol(broker) == common

    def test_unsupported(self):
        with pytest.raises(ValueError):
            self.mapper.to_broker_symbol("FOO/BAR") 