"""
Unit tests for the project-wide types system.
"""

import pytest
import sys
from pathlib import Path
from enum import Enum
import pandas as pd
import numpy as np
import inspect
from collections import Counter
import unittest
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.types.enums import (
    BrokerType, AssetClass, Timeframe, OrderType, OrderSide, OrderStatus,
    MarketSession, EventImportance, StrategyType, SignalType, RiskLevel
)
from src.types.common import (
    SymbolType, PriceType, VolumeType, TimestampType, OHLCVData, TradingSession
)
import src.types as types_module


@pytest.mark.unit
class TestTypesFileStructure:
    """Test the structure and integrity of the types files."""
    
    def test_enum_classes_properly_importable(self):
        """Test that all enum classes can be imported and are proper Enums."""
        enum_classes = [
            BrokerType, AssetClass, Timeframe, OrderType, OrderSide, OrderStatus,
            MarketSession, EventImportance, StrategyType, SignalType, RiskLevel
        ]
        
        for enum_class in enum_classes:
            assert issubclass(enum_class, Enum), f"{enum_class.__name__} is not an Enum subclass"
            assert len(enum_class) > 0, f"{enum_class.__name__} has no values"
            for member in enum_class:
                assert isinstance(member.value, str), f"{enum_class.__name__}.{member.name} value is not a string"
    
    def test_type_aliases_structure(self):
        """Test that type aliases are properly defined and accessible."""
        # Test that we can access all the imported type aliases
        type_aliases = [
            'SymbolType', 'PriceType', 'VolumeType', 'TimestampType',
            'OHLCVData', 'FeatureData', 'IndicatorData', 'SignalData',
            'ValidationResult', 'ErrorContext'
        ]
        
        for alias_name in type_aliases:
            assert hasattr(types_module, alias_name), f"Type alias {alias_name} not found in types module"
    
    def test_broker_type_single_definition(self):
        """Specifically test that BrokerType is defined only once."""
        # Get all attributes from the types module
        broker_type_attrs = [name for name in dir(types_module) if name == 'BrokerType']
        
        # Should be exactly one
        assert len(broker_type_attrs) == 1, f"Expected exactly one BrokerType, found {len(broker_type_attrs)}"
        
        # Should be the correct class
        broker_type = getattr(types_module, 'BrokerType')
        assert issubclass(broker_type, Enum), "BrokerType should be an Enum subclass"
        
        # Check expected values exist (updated to only include currently supported brokers)
        expected_values = {'FOREX_COM', 'GENERIC'}
        actual_values = {member.name for member in broker_type}
        
        assert expected_values == actual_values, f"BrokerType values mismatch. Expected: {expected_values}, Got: {actual_values}"


@pytest.mark.unit
class TestCoreTypes:
    """Test core trading type aliases."""
    
    def test_symbol_type(self):
        """Test SymbolType alias."""
        symbol: SymbolType = "EUR/USD"
        assert isinstance(symbol, str)
        assert symbol == "EUR/USD"
    
    def test_price_type(self):
        """Test PriceType alias."""
        price: PriceType = 1.0985
        assert isinstance(price, float)
        assert price == 1.0985
        
        # Should also accept int
        price_int: PriceType = 100
        assert isinstance(price_int, (int, float))
    
    def test_volume_type(self):
        """Test VolumeType alias."""
        volume: VolumeType = 10000.0
        assert isinstance(volume, (int, float))
        
        volume_int: VolumeType = 5000
        assert isinstance(volume_int, (int, float))
    
    def test_timestamp_type(self):
        """Test TimestampType alias."""
        timestamp: TimestampType = datetime.now()
        assert isinstance(timestamp, datetime)


@pytest.mark.unit
class TestDataStructureTypes:
    """Test data structure type aliases."""
    
    def test_ohlcv_data(self):
        """Test OHLCVData type alias."""
        data = pd.DataFrame({'open': [1.0], 'high': [1.1], 'low': [0.9], 'close': [1.05], 'volume': [100]})
        ohlcv: OHLCVData = data
        assert isinstance(ohlcv, pd.DataFrame)
    
    def test_trading_session_type(self):
        """Test the TradingSession TypeAlias."""
        session: TradingSession = {"name": "London", "status": "open"}
        assert isinstance(session, dict)


@pytest.mark.unit
class TestTimeframe:
    """Test Timeframe enum."""
    
    def test_timeframe_minutes_property(self):
        """Test timeframe minutes conversion."""
        assert Timeframe.H1.minutes == 60
        assert Timeframe.D1.minutes == 1440
    
    def test_timeframe_from_minutes(self):
        """Test creating timeframe from minutes."""
        assert Timeframe.from_minutes(60) == Timeframe.H1
        with pytest.raises(ValueError):
            Timeframe.from_minutes(7)


@pytest.mark.unit
class TestTypeIntegration:
    """Test type system integration and consistency."""
    
    def test_enum_uniqueness(self):
        """Test that enum values are unique within each enum."""
        enums_to_test = [
            BrokerType, AssetClass, Timeframe, OrderType, OrderSide, 
            OrderStatus, MarketSession, EventImportance,
            StrategyType, SignalType, RiskLevel
        ]
        
        for enum_class in enums_to_test:
            values = [member.value for member in enum_class]
            assert len(values) == len(set(values)), f"Duplicate values in {enum_class.__name__}"
    
    def test_enum_naming_conventions(self):
        """Test that enum values follow naming conventions."""
        enums_to_test = [
            BrokerType, AssetClass, OrderType, OrderSide, 
            OrderStatus, MarketSession, EventImportance,
            StrategyType, SignalType, RiskLevel
        ]
        
        for enum_class in enums_to_test:
            for member in enum_class:
                assert member.value.islower() or '_' in member.value, \
                    f"Invalid naming in {enum_class.__name__}.{member.name}: {member.value}"


if __name__ == "__main__":
    pytest.main([__file__]) 