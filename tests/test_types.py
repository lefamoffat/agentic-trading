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

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.types import (
    # Core trading types
    SymbolType, PriceType, VolumeType, TimestampType,
    
    # Data structure types
    OHLCVData, FeatureData, IndicatorData, SignalData,
    TradingSession, AccountInfo, Position, Order,
    
    # Enums
    BrokerType, AssetClass, Timeframe, OrderType, OrderSide, OrderStatus,
    IndicatorType, MarketSession, EventImportance, StrategyType,
    SignalType, RiskLevel,
    
    # Function types
    IndicatorFunction, ValidationFunction, StrategyFunction,
    
    # Validation types
    ValidationResult, ErrorContext
)

import src.types as types_module


class TestTypesFileStructure:
    """Test the structure and integrity of the types file."""
    
    def test_no_duplicate_class_definitions(self):
        """Test that there are no duplicate class definitions in types.py."""
        types_file = Path(__file__).parent.parent / "src" / "types.py"
        with open(types_file, 'r') as f:
            content = f.read()
        
        # Extract all class definitions
        lines = content.split('\n')
        class_definitions = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('class ') and '(' in stripped:
                # Extract class name
                class_name = stripped.split('class ')[1].split('(')[0].strip()
                class_definitions.append((class_name, i + 1))  # +1 for 1-indexed line numbers
        
        # Check for duplicates
        class_names = [name for name, _ in class_definitions]
        name_counts = Counter(class_names)
        
        duplicates = {name: count for name, count in name_counts.items() if count > 1}
        
        if duplicates:
            duplicate_details = []
            for name in duplicates:
                lines_with_class = [line_num for class_name, line_num in class_definitions if class_name == name]
                duplicate_details.append(f"{name} defined on lines: {lines_with_class}")
            
            pytest.fail(f"Found duplicate class definitions:\n" + "\n".join(duplicate_details))
        
        # Ensure we found the expected classes
        expected_classes = {
            'BrokerType', 'AssetClass', 'Timeframe', 'OrderType', 'OrderSide', 
            'OrderStatus', 'DataQuality', 'IndicatorType', 'MarketSession', 'TradingSession',
            'EventImportance', 'StrategyType', 'SignalType', 'RiskLevel'
        }
        
        found_classes = set(class_names)
        assert expected_classes.issubset(found_classes), f"Missing expected classes: {expected_classes - found_classes}"
    
    def test_enum_classes_properly_importable(self):
        """Test that all enum classes can be imported and are proper Enums."""
        enum_classes = [
            BrokerType, AssetClass, Timeframe, OrderType, OrderSide, OrderStatus,
            IndicatorType, MarketSession, TradingSession, EventImportance, StrategyType, SignalType, RiskLevel
        ]
        
        for enum_class in enum_classes:
            # Check it's actually an Enum
            assert issubclass(enum_class, Enum), f"{enum_class.__name__} is not an Enum subclass"
            
            # Check it has values
            assert len(enum_class) > 0, f"{enum_class.__name__} has no values"
            
            # Check all values are strings
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
        from datetime import datetime
        
        timestamp: TimestampType = datetime.now()
        assert isinstance(timestamp, datetime)


class TestDataStructureTypes:
    """Test data structure type aliases."""
    
    def test_ohlcv_data(self):
        """Test OHLCVData type alias."""
        data = pd.DataFrame({
            'open': [1.0980, 1.0985],
            'high': [1.0990, 1.0995],
            'low': [1.0975, 1.0980],
            'close': [1.0985, 1.0990],
            'volume': [1000, 1500]
        })
        
        ohlcv: OHLCVData = data
        assert isinstance(ohlcv, pd.DataFrame)
        assert all(col in ohlcv.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    def test_feature_data(self):
        """Test FeatureData type alias."""
        features = pd.DataFrame({
            'sma_20': [1.0980, 1.0985],
            'rsi_14': [65.5, 68.2],
            'macd_signal': [0.0012, 0.0015]
        })
        
        feature_data: FeatureData = features
        assert isinstance(feature_data, pd.DataFrame)


class TestBrokerType:
    """Test BrokerType enum."""
    
    def test_broker_type_values(self):
        """Test all broker type values."""
        assert BrokerType.FOREX_COM.value == "forex_com"
        assert BrokerType.GENERIC.value == "generic"
    
    def test_broker_type_enumeration(self):
        """Test broker type is proper enum."""
        assert isinstance(BrokerType.FOREX_COM, BrokerType)
        assert isinstance(BrokerType, type)
        assert issubclass(BrokerType, Enum)
    
    def test_broker_type_iteration(self):
        """Test iterating over broker types."""
        broker_values = [broker.value for broker in BrokerType]
        expected = ["forex_com", "generic"]
        assert all(value in broker_values for value in expected)
        assert len(broker_values) == len(expected)


class TestAssetClass:
    """Test AssetClass enum."""
    
    def test_asset_class_values(self):
        """Test asset class values."""
        assert AssetClass.FOREX.value == "forex"
        assert AssetClass.STOCKS.value == "stocks"
        assert AssetClass.CRYPTO.value == "crypto"
        assert AssetClass.COMMODITIES.value == "commodities"
        assert AssetClass.INDICES.value == "indices"
        assert AssetClass.BONDS.value == "bonds"
    
    def test_asset_class_enumeration(self):
        """Test asset class is proper enum."""
        assert isinstance(AssetClass.FOREX, AssetClass)
        assert issubclass(AssetClass, Enum)


class TestTimeframe:
    """Test Timeframe enum."""
    
    def test_timeframe_values(self):
        """Test timeframe values."""
        assert Timeframe.M1.value == "1m"
        assert Timeframe.M5.value == "5m"
        assert Timeframe.M15.value == "15m"
        assert Timeframe.H1.value == "1h"
        assert Timeframe.H4.value == "4h"
        assert Timeframe.D1.value == "1d"
        assert Timeframe.W1.value == "1w"
        assert Timeframe.MN1.value == "1M"
    
    def test_timeframe_minutes_property(self):
        """Test timeframe minutes conversion."""
        assert Timeframe.M1.minutes == 1
        assert Timeframe.M5.minutes == 5
        assert Timeframe.M15.minutes == 15
        assert Timeframe.H1.minutes == 60
        assert Timeframe.H4.minutes == 240
        assert Timeframe.D1.minutes == 1440
    
    def test_timeframe_from_minutes(self):
        """Test creating timeframe from minutes."""
        assert Timeframe.from_minutes(1) == Timeframe.M1
        assert Timeframe.from_minutes(5) == Timeframe.M5
        assert Timeframe.from_minutes(60) == Timeframe.H1
        assert Timeframe.from_minutes(1440) == Timeframe.D1
    
    def test_timeframe_from_minutes_invalid(self):
        """Test invalid minutes raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            Timeframe.from_minutes(7)  # Invalid timeframe


class TestOrderTypes:
    """Test order-related enums."""
    
    def test_order_type_values(self):
        """Test OrderType values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        assert OrderType.STOP_LIMIT.value == "stop_limit"
        assert OrderType.TRAILING_STOP.value == "trailing_stop"
    
    def test_order_side_values(self):
        """Test OrderSide values."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"
    
    def test_order_status_values(self):
        """Test OrderStatus values."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"
        assert OrderStatus.PARTIALLY_FILLED.value == "partially_filled"


class TestIndicatorType:
    """Test IndicatorType enum."""
    
    def test_trend_indicators(self):
        """Test trend indicator types."""
        trend_indicators = [
            IndicatorType.SMA, IndicatorType.EMA, IndicatorType.WMA,
            IndicatorType.DEMA, IndicatorType.TEMA, IndicatorType.TRIMA,
            IndicatorType.KAMA, IndicatorType.MAMA, IndicatorType.T3
        ]
        
        for indicator in trend_indicators:
            assert isinstance(indicator, IndicatorType)
            assert isinstance(indicator.value, str)
    
    def test_momentum_indicators(self):
        """Test momentum indicator types."""
        momentum_indicators = [
            IndicatorType.RSI, IndicatorType.STOCH, IndicatorType.STOCH_RSI,
            IndicatorType.MACD, IndicatorType.ADX, IndicatorType.CCI,
            IndicatorType.MFI, IndicatorType.WILLIAMS_R, IndicatorType.ROC,
            IndicatorType.CMO
        ]
        
        for indicator in momentum_indicators:
            assert isinstance(indicator, IndicatorType)
            assert indicator.value in [
                "rsi", "stoch", "stoch_rsi", "macd", "adx", "cci",
                "mfi", "williams_r", "roc", "cmo"
            ]
    
    def test_volatility_indicators(self):
        """Test volatility indicator types."""
        volatility_indicators = [
            IndicatorType.BOLLINGER_BANDS, IndicatorType.ATR, IndicatorType.NATR,
            IndicatorType.TRANGE
        ]
        
        for indicator in volatility_indicators:
            assert isinstance(indicator, IndicatorType)
    
    def test_volume_indicators(self):
        """Test volume indicator types."""
        volume_indicators = [
            IndicatorType.AD, IndicatorType.ADOSC, IndicatorType.OBV
        ]
        
        for indicator in volume_indicators:
            assert isinstance(indicator, IndicatorType)
    
    def test_all_indicators_have_values(self):
        """Test all indicators have proper string values."""
        for indicator in IndicatorType:
            assert isinstance(indicator.value, str)
            assert len(indicator.value) > 0
            assert indicator.value.islower() or '_' in indicator.value


class TestMarketSession:
    """Test MarketSession enum."""
    
    def test_market_session_values(self):
        """Test market session values."""
        assert MarketSession.CLOSED.value == "closed"
        assert MarketSession.OPEN.value == "open"
        assert MarketSession.PRE_MARKET.value == "pre_market"
        assert MarketSession.POST_MARKET.value == "post_market"
        assert MarketSession.OVERLAP.value == "overlap"


class TestTradingSession:
    """Test TradingSession enum."""
    
    def test_trading_session_values(self):
        """Test trading session values."""
        assert TradingSession.LONDON.value == "london"
        assert TradingSession.NEW_YORK.value == "new_york"
        assert TradingSession.TOKYO.value == "tokyo"
        assert TradingSession.SYDNEY.value == "sydney"


class TestEventImportance:
    """Test EventImportance enum."""
    
    def test_event_importance_values(self):
        """Test event importance values."""
        assert EventImportance.LOW.value == "low"
        assert EventImportance.MEDIUM.value == "medium"
        assert EventImportance.HIGH.value == "high"
        assert EventImportance.CRITICAL.value == "critical"
    
    def test_event_importance_ordering(self):
        """Test event importance can be compared."""
        # Since these are string values, we'll test they exist rather than ordering
        importance_levels = [e.value for e in EventImportance]
        expected_levels = ["low", "medium", "high", "critical"]
        assert all(level in importance_levels for level in expected_levels)


class TestStrategyTypes:
    """Test strategy-related enums."""
    
    def test_strategy_type_values(self):
        """Test StrategyType values."""
        assert StrategyType.TREND_FOLLOWING.value == "trend_following"
        assert StrategyType.MEAN_REVERSION.value == "mean_reversion"
        assert StrategyType.MOMENTUM.value == "momentum"
        assert StrategyType.ARBITRAGE.value == "arbitrage"
        assert StrategyType.SCALPING.value == "scalping"
        assert StrategyType.SWING.value == "swing"
        assert StrategyType.POSITION.value == "position"
    
    def test_signal_type_values(self):
        """Test SignalType values."""
        assert SignalType.BUY.value == "buy"
        assert SignalType.SELL.value == "sell"
        assert SignalType.HOLD.value == "hold"
        assert SignalType.CLOSE.value == "close"
    
    def test_risk_level_values(self):
        """Test RiskLevel values."""
        assert RiskLevel.VERY_LOW.value == "very_low"
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.VERY_HIGH.value == "very_high"


class TestFunctionTypes:
    """Test function type aliases."""
    
    def test_indicator_function_type(self):
        """Test IndicatorFunction type alias."""
        def sample_indicator(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
            return data
        
        func: IndicatorFunction = sample_indicator
        assert callable(func)
    
    def test_validation_function_type(self):
        """Test ValidationFunction type alias."""
        def sample_validator(value, **kwargs) -> bool:
            return True
        
        func: ValidationFunction = sample_validator
        assert callable(func)
    
    def test_strategy_function_type(self):
        """Test StrategyFunction type alias."""
        def sample_strategy(data: pd.DataFrame) -> pd.Series:
            return pd.Series([0] * len(data))
        
        func: StrategyFunction = sample_strategy
        assert callable(func)


class TestValidationTypes:
    """Test validation-related types."""
    
    def test_validation_result_structure(self):
        """Test ValidationResult type structure."""
        # Valid result
        valid_result: ValidationResult = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        assert valid_result['is_valid'] is True
        assert isinstance(valid_result['errors'], list)
        assert isinstance(valid_result['warnings'], list)
        
        # Invalid result with errors
        invalid_result: ValidationResult = {
            'is_valid': False,
            'errors': ['Missing required column: close'],
            'warnings': ['Data contains gaps']
        }
        
        assert invalid_result['is_valid'] is False
        assert len(invalid_result['errors']) == 1
        assert len(invalid_result['warnings']) == 1
    
    def test_error_context_structure(self):
        """Test ErrorContext type structure."""
        context: ErrorContext = {
            'operation': 'indicator_calculation',
            'symbol': 'EUR/USD',
            'timeframe': '1h',
            'parameters': {'period': 14},
            'data_shape': (100, 5)
        }
        
        assert context['operation'] == 'indicator_calculation'
        assert context['symbol'] == 'EUR/USD'
        assert context['timeframe'] == '1h'
        assert isinstance(context['parameters'], dict)
        assert isinstance(context['data_shape'], tuple)


class TestTypeIntegration:
    """Test type system integration and consistency."""
    
    def test_enum_uniqueness(self):
        """Test that enum values are unique within each enum."""
        enums_to_test = [
            BrokerType, AssetClass, Timeframe, OrderType, OrderSide, 
            OrderStatus, IndicatorType, MarketSession, TradingSession, EventImportance,
            StrategyType, SignalType, RiskLevel
        ]
        
        for enum_class in enums_to_test:
            values = [member.value for member in enum_class]
            assert len(values) == len(set(values)), f"Duplicate values in {enum_class.__name__}"
    
    def test_enum_naming_conventions(self):
        """Test that enum values follow naming conventions."""
        enums_to_test = [
            BrokerType, AssetClass, OrderType, OrderSide,
            OrderStatus, IndicatorType, MarketSession, EventImportance,
            StrategyType, SignalType, RiskLevel
        ]
        
        for enum_class in enums_to_test:
            for member in enum_class:
                # Values should be lowercase with underscores
                assert member.value.islower() or '_' in member.value, \
                    f"Invalid naming in {enum_class.__name__}.{member.name}: {member.value}"
        
        # Timeframe has special case values like "1M", "1h", etc.
        for member in Timeframe:
            # Timeframe values can contain numbers and letters
            assert isinstance(member.value, str) and len(member.value) > 0, \
                f"Invalid timeframe value: {member.value}"
    
    def test_comprehensive_indicator_coverage(self):
        """Test that we have comprehensive indicator type coverage."""
        # Should have at least 25+ indicators as mentioned in requirements
        assert len(IndicatorType) >= 25
        
        # Should cover major categories
        trend_count = len([i for i in IndicatorType if i.value in [
            'sma', 'ema', 'wma', 'dema', 'tema', 'trima', 'kama', 'mama', 't3'
        ]])
        momentum_count = len([i for i in IndicatorType if i.value in [
            'rsi', 'stoch', 'stoch_rsi', 'macd', 'adx', 'cci', 'mfi', 'williams_r', 'roc', 'cmo'
        ]])
        volatility_count = len([i for i in IndicatorType if i.value in [
            'bbands', 'atr', 'natr', 'trange'
        ]])
        volume_count = len([i for i in IndicatorType if i.value in [
            'ad', 'adosc', 'obv'
        ]])
        
        assert trend_count >= 6, "Should have at least 6 trend indicators"
        assert momentum_count >= 8, "Should have at least 8 momentum indicators"
        assert volatility_count >= 3, "Should have at least 3 volatility indicators"
        assert volume_count >= 3, "Should have at least 3 volume indicators"


if __name__ == "__main__":
    pytest.main([__file__]) 