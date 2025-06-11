"""
Unit tests for the SymbolMapper class.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.brokers.symbol_mapper import SymbolMapper, BrokerType


class TestSymbolMapper:
    """Test cases for SymbolMapper class."""
    
    def test_forex_com_initialization(self):
        """Test SymbolMapper initialization for Forex.com."""
        mapper = SymbolMapper(BrokerType.FOREX_COM)
        assert mapper.broker_type == BrokerType.FOREX_COM
        assert len(mapper.to_broker_mappings) > 0
        assert len(mapper.from_broker_mappings) > 0
    
    def test_eur_usd_mapping(self):
        """Test EUR/USD symbol mapping."""
        mapper = SymbolMapper(BrokerType.FOREX_COM)
        
        broker_symbol = mapper.to_broker_symbol("EUR/USD")
        assert broker_symbol == "EUR_USD"
        
        common_symbol = mapper.from_broker_symbol("EUR_USD")
        assert common_symbol == "EUR/USD"
        
        roundtrip = mapper.from_broker_symbol(mapper.to_broker_symbol("EUR/USD"))
        assert roundtrip == "EUR/USD"
    
    def test_gbp_usd_mapping(self):
        """Test GBP/USD symbol mapping."""
        mapper = SymbolMapper(BrokerType.FOREX_COM)
        
        broker_symbol = mapper.to_broker_symbol("GBP/USD")
        assert broker_symbol == "GBP_USD"
        
        common_symbol = mapper.from_broker_symbol("GBP_USD")
        assert common_symbol == "GBP/USD"
    
    def test_major_pairs_mapping(self):
        """Test mapping for all major forex pairs."""
        mapper = SymbolMapper(BrokerType.FOREX_COM)
        
        major_pairs = [
            ("EUR/USD", "EUR_USD"),
            ("GBP/USD", "GBP_USD"),
            ("USD/JPY", "USD_JPY"),
            ("USD/CHF", "USD_CHF"),
            ("USD/CAD", "USD_CAD"),
            ("AUD/USD", "AUD_USD"),
            ("NZD/USD", "NZD_USD")
        ]
        
        for common, expected_broker in major_pairs:
            broker_symbol = mapper.to_broker_symbol(common)
            assert broker_symbol == expected_broker
            
            back_to_common = mapper.from_broker_symbol(broker_symbol)
            assert back_to_common == common
    
    def test_unsupported_symbol_to_broker(self):
        """Test that unsupported symbols raise ValueError when converting to broker format."""
        mapper = SymbolMapper(BrokerType.FOREX_COM)
        
        with pytest.raises(ValueError, match="not supported"):
            mapper.to_broker_symbol("UNSUPPORTED/PAIR")
    
    def test_unsupported_symbol_from_broker(self):
        """Test that unsupported broker symbols raise ValueError when converting to common format."""
        mapper = SymbolMapper(BrokerType.FOREX_COM)
        
        with pytest.raises(ValueError, match="not recognized"):
            mapper.from_broker_symbol("UNSUPPORTED_PAIR")
    
    def test_is_supported(self):
        """Test is_supported method."""
        mapper = SymbolMapper(BrokerType.FOREX_COM)
        
        assert mapper.is_supported("EUR/USD") is True
        assert mapper.is_supported("GBP/USD") is True
        assert mapper.is_supported("UNSUPPORTED/PAIR") is False
    
    def test_get_supported_symbols(self):
        """Test get_supported_symbols method."""
        mapper = SymbolMapper(BrokerType.FOREX_COM)
        
        supported = mapper.get_supported_symbols()
        assert isinstance(supported, list)
        assert len(supported) > 20  # Should have many pairs
        assert "EUR/USD" in supported
        assert "GBP/USD" in supported
        assert "USD/JPY" in supported
    
    def test_generic_mapper(self):
        """Test generic mapper (no mappings)."""
        mapper = SymbolMapper(BrokerType.GENERIC)
        
        # Generic mapper should pass through symbols unchanged
        assert mapper.to_broker_symbol("EUR/USD") == "EUR/USD"
        assert mapper.from_broker_symbol("EUR_USD") == "EUR_USD"
        assert mapper.is_supported("ANYTHING") is True
        assert mapper.get_supported_symbols() == []
    
    def test_add_custom_mapping(self):
        """Test adding custom mappings."""
        # Add a custom mapping
        SymbolMapper.add_custom_mapping(
            BrokerType.FOREX_COM,
            "CUSTOM/PAIR",
            "CUSTOM_PAIR"
        )
        
        # Test that it works
        mapper = SymbolMapper(BrokerType.FOREX_COM)
        assert mapper.to_broker_symbol("CUSTOM/PAIR") == "CUSTOM_PAIR"
        assert mapper.from_broker_symbol("CUSTOM_PAIR") == "CUSTOM/PAIR"
        
        # Clean up by removing the custom mapping
        del SymbolMapper.FOREX_COM_MAPPINGS["CUSTOM/PAIR"]
        del SymbolMapper.FOREX_COM_REVERSE_MAPPINGS["CUSTOM_PAIR"]
    
    def test_alpaca_mappings_exist(self):
        """Test that Alpaca mappings are defined (for future use)."""
        mapper = SymbolMapper(BrokerType.ALPACA)
        
        # Should have some basic mappings defined
        assert len(mapper.to_broker_mappings) > 0
        assert "EUR/USD" in mapper.to_broker_mappings
        assert mapper.to_broker_symbol("EUR/USD") == "EURUSD"  # Alpaca format


if __name__ == "__main__":
    pytest.main([__file__]) 