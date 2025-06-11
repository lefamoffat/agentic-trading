#!/usr/bin/env python3
"""
Test script for Phase 3: Feature Engineering Framework.

Tests the complete feature engineering pipeline including:
- Technical indicators calculation
- Feature pipeline orchestration
- Data validation and preprocessing
- Feature selection and metadata generation
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features import (
    FeaturePipeline, PipelineConfig, generate_features,
    FeatureFactory, IndicatorType, FeatureCalculator,
    MovingAverage, MACD, RSI, BollingerBands
)

# Handle logger import with fallback
try:
    from utils.logger import get_logger
except ImportError:
    import logging
    def get_logger(name):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)


def generate_sample_data(periods: int = 1000) -> pd.DataFrame:
    """Generate realistic sample OHLCV data for testing."""
    
    # Start from a base price
    start_price = 1.2000  # EUR/USD typical
    
    # Generate random walk with volatility clustering
    np.random.seed(42)  # For reproducible results
    
    # Price changes with some trend and volatility clustering
    returns = np.random.normal(0, 0.001, periods)  # 0.1% daily volatility
    returns[100:200] *= 2  # Volatility cluster
    returns[500:600] += 0.0005  # Trend period
    
    # Generate prices
    prices = start_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from prices
    data = []
    for i, close in enumerate(prices):
        if i == 0:
            open_price = start_price
        else:
            open_price = prices[i-1]
        
        # Add some intraday volatility
        daily_vol = abs(returns[i]) * 2
        high = max(open_price, close) + np.random.uniform(0, daily_vol * close)
        low = min(open_price, close) - np.random.uniform(0, daily_vol * close)
        
        # Ensure proper OHLC relationships
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate volume (higher volume during volatile periods)
        base_volume = 100000
        volume = base_volume * (1 + 5 * abs(returns[i]) + np.random.uniform(0.5, 1.5))
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': int(volume)
        })
    
    # Create DataFrame with datetime index
    df = pd.DataFrame(data)
    
    # Generate datetime index (hourly data)
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(periods)]
    df.index = pd.DatetimeIndex(dates)
    
    return df


def test_individual_indicators():
    """Test individual technical indicators."""
    logger = get_logger("test_indicators")
    logger.info("Testing individual technical indicators...")
    
    # Generate test data
    data = generate_sample_data(500)
    
    # Test Moving Average
    sma = MovingAverage(period=20)
    sma_result = sma.apply(data)
    assert not sma_result.empty
    assert 'sma_20' in sma_result.columns
    logger.info(f"✓ SMA calculation successful: {sma_result.shape}")
    
    # Test MACD
    macd = MACD()
    macd_result = macd.apply(data)
    assert not macd_result.empty
    assert 'macd_line' in macd_result.columns
    assert 'macd_signal' in macd_result.columns
    assert 'macd_histogram' in macd_result.columns
    logger.info(f"✓ MACD calculation successful: {macd_result.shape}")
    
    # Test RSI
    rsi = RSI(period=14)
    rsi_result = rsi.apply(data)
    assert not rsi_result.empty
    assert 'rsi_14' in rsi_result.columns
    # Check RSI bounds
    rsi_values = rsi_result['rsi_14'].dropna()
    assert rsi_values.min() >= 0
    assert rsi_values.max() <= 100
    logger.info(f"✓ RSI calculation successful: {rsi_result.shape}")
    
    # Test Bollinger Bands
    bb = BollingerBands(period=20, std_dev=2)
    bb_result = bb.apply(data)
    assert not bb_result.empty
    assert 'bb_upper' in bb_result.columns
    assert 'bb_lower' in bb_result.columns
    assert 'bb_middle' in bb_result.columns
    logger.info(f"✓ Bollinger Bands calculation successful: {bb_result.shape}")
    
    logger.info("All individual indicator tests passed!")


def test_feature_factory():
    """Test the feature factory functionality."""
    logger = get_logger("test_factory")
    logger.info("Testing feature factory...")
    
    factory = FeatureFactory()
    
    # Test creating individual indicators
    sma = factory.create_indicator(IndicatorType.SMA, period=50)
    assert isinstance(sma, MovingAverage)
    assert sma.period == 50
    logger.info("✓ Individual indicator creation successful")
    
    # Test creating common indicators
    common_indicators = factory.get_common_indicators()
    assert len(common_indicators) > 0
    logger.info(f"✓ Common indicators created: {len(common_indicators)}")
    
    # Test creating forex indicators
    forex_indicators = factory.get_forex_indicators()
    assert len(forex_indicators) > 0
    logger.info(f"✓ Forex indicators created: {len(forex_indicators)}")
    
    # Test creating scalping indicators
    scalping_indicators = factory.get_scalping_indicators()
    assert len(scalping_indicators) > 0
    logger.info(f"✓ Scalping indicators created: {len(scalping_indicators)}")
    
    # Test parameter overrides
    custom_indicators = factory.get_common_indicators(
        simple_moving_average_period=30
    )
    logger.info("✓ Parameter override test successful")
    
    logger.info("All feature factory tests passed!")


def test_feature_calculator():
    """Test the feature calculator."""
    logger = get_logger("test_calculator")
    logger.info("Testing feature calculator...")
    
    # Generate test data
    data = generate_sample_data(200)
    
    # Create a basic config for the calculator
    from features.calculator import FeatureConfig
    basic_config = FeatureConfig(
        indicators=[],  # No indicators for basic test
        include_price_features=True,
        include_time_features=True
    )
    
    calculator = FeatureCalculator(basic_config)
    
    # Test price features
    config = FeatureConfig(
        indicators=[],
        include_price_features=True,
        include_time_features=False
    )
    
    features = calculator.calculate_features(data)
    assert not features.empty
    assert len(features.columns) > 0
    logger.info(f"✓ Price features calculation: {features.shape}")
    
    # Test time features
    config = FeatureConfig(
        indicators=[],
        include_price_features=False,
        include_time_features=True
    )
    
    calculator_time = FeatureCalculator(config)
    time_features = calculator_time.calculate_features(data)
    assert not time_features.empty
    logger.info(f"✓ Time features calculation: {time_features.shape}")
    
    logger.info("All feature calculator tests passed!")


def test_feature_pipeline():
    """Test the complete feature pipeline."""
    logger = get_logger("test_pipeline")
    logger.info("Testing complete feature pipeline...")
    
    # Generate test data
    data = generate_sample_data(500)
    
    # Test with default configuration
    config = PipelineConfig(
        min_periods=50,
        feature_selection_enabled=True
    )
    
    pipeline = FeaturePipeline(config)
    features = pipeline.transform(data)
    
    assert not features.empty
    assert len(features.columns) > 10  # Should have many features
    logger.info(f"✓ Complete pipeline transformation: {features.shape}")
    
    # Test metadata generation
    metadata = pipeline.get_metadata()
    assert 'pipeline_config' in metadata
    assert 'data_info' in metadata
    assert 'indicators' in metadata
    logger.info("✓ Metadata generation successful")
    
    # Test feature names
    feature_names = pipeline.get_feature_names()
    assert len(feature_names) == len(features.columns)
    logger.info(f"✓ Feature names: {len(feature_names)} features")
    
    # Test forex-optimized pipeline
    forex_config = PipelineConfig(
        use_forex_optimized=True,
        use_scalping_indicators=False,
        min_periods=50
    )
    
    forex_pipeline = FeaturePipeline(forex_config)
    forex_features = forex_pipeline.transform(data)
    assert not forex_features.empty
    logger.info(f"✓ Forex-optimized pipeline: {forex_features.shape}")
    
    # Test scalping pipeline
    scalping_config = PipelineConfig(
        use_forex_optimized=False,
        use_scalping_indicators=True,
        min_periods=20
    )
    
    scalping_pipeline = FeaturePipeline(scalping_config)
    scalping_features = scalping_pipeline.transform(data)
    assert not scalping_features.empty
    logger.info(f"✓ Scalping pipeline: {scalping_features.shape}")
    
    logger.info("All feature pipeline tests passed!")


def test_convenience_function():
    """Test the convenience generate_features function."""
    logger = get_logger("test_convenience")
    logger.info("Testing convenience function...")
    
    # Generate test data
    data = generate_sample_data(300)
    
    # Test with defaults
    features = generate_features(data)
    assert not features.empty
    logger.info(f"✓ Default generation: {features.shape}")
    
    # Test with custom parameters
    features_custom = generate_features(
        data,
        min_periods=50,
        use_forex_optimized=True,
        feature_selection_enabled=True
    )
    assert not features_custom.empty
    logger.info(f"✓ Custom parameter generation: {features_custom.shape}")
    
    logger.info("Convenience function tests passed!")


def test_data_validation():
    """Test data validation functionality."""
    logger = get_logger("test_validation")
    logger.info("Testing data validation...")
    
    pipeline = FeaturePipeline()
    
    # Test with valid data
    valid_data = generate_sample_data(200)
    is_valid, errors = pipeline.validate_data(valid_data)
    assert is_valid
    assert len(errors) == 0
    logger.info("✓ Valid data validation passed")
    
    # Test with insufficient data
    small_data = generate_sample_data(50)
    is_valid, errors = pipeline.validate_data(small_data)
    assert not is_valid
    assert any("Insufficient data" in error for error in errors)
    logger.info("✓ Insufficient data validation caught")
    
    # Test with missing columns
    incomplete_data = valid_data.drop(columns=['close'])
    is_valid, errors = pipeline.validate_data(incomplete_data)
    assert not is_valid
    assert any("Missing required columns" in error for error in errors)
    logger.info("✓ Missing columns validation caught")
    
    # Test with invalid OHLC
    invalid_data = valid_data.copy()
    invalid_data.loc[invalid_data.index[10], 'high'] = invalid_data.loc[invalid_data.index[10], 'low'] - 0.001
    is_valid, errors = pipeline.validate_data(invalid_data)
    assert not is_valid
    assert any("Invalid OHLC" in error for error in errors)
    logger.info("✓ Invalid OHLC validation caught")
    
    logger.info("All data validation tests passed!")


def test_with_real_sample():
    """Test with a more realistic sample that mimics actual forex data."""
    logger = get_logger("test_real_sample")
    logger.info("Testing with realistic forex sample...")
    
    # Create more realistic EUR/USD sample
    np.random.seed(123)
    periods = 2000  # About 83 days of hourly data
    
    # More realistic EUR/USD parameters
    start_price = 1.1850
    daily_vol = 0.0080  # 0.8% daily volatility
    hourly_vol = daily_vol / np.sqrt(24)
    
    # Add some market session effects
    returns = []
    for i in range(periods):
        hour = i % 24
        
        # Higher volatility during London/NY overlap
        if 13 <= hour <= 17:  # London-NY overlap
            vol_mult = 1.5
        elif 8 <= hour <= 17:  # London session
            vol_mult = 1.2
        elif 13 <= hour <= 22:  # NY session
            vol_mult = 1.1
        else:  # Asian session
            vol_mult = 0.7
        
        # Add weekly pattern (lower volatility on weekends)
        day_of_week = (i // 24) % 7
        if day_of_week >= 5:  # Weekend
            vol_mult *= 0.3
        
        return_val = np.random.normal(0, hourly_vol * vol_mult)
        returns.append(return_val)
    
    # Generate prices
    prices = start_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    data = []
    for i, close in enumerate(prices):
        if i == 0:
            open_price = start_price
        else:
            open_price = prices[i-1]
        
        # More realistic intraday movements
        price_range = abs(open_price - close)
        extra_range = price_range * np.random.uniform(0.2, 0.8)
        
        high = max(open_price, close) + extra_range
        low = min(open_price, close) - extra_range
        
        # Typical forex volume patterns
        hour = i % 24
        base_volume = 1000000
        
        if 13 <= hour <= 17:  # London-NY overlap
            volume = base_volume * np.random.uniform(1.5, 3.0)
        elif 8 <= hour <= 17:  # London session
            volume = base_volume * np.random.uniform(1.0, 2.0)
        elif 13 <= hour <= 22:  # NY session
            volume = base_volume * np.random.uniform(0.8, 1.5)
        else:  # Asian session
            volume = base_volume * np.random.uniform(0.3, 0.8)
        
        data.append({
            'open': round(open_price, 5),
            'high': round(high, 5),
            'low': round(low, 5),
            'close': round(close, 5),
            'volume': int(volume)
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    start_date = datetime(2024, 1, 1, 0, 0, 0)
    dates = [start_date + timedelta(hours=i) for i in range(periods)]
    df.index = pd.DatetimeIndex(dates)
    
    logger.info(f"Generated realistic sample: {df.shape}")
    logger.info(f"Price range: {df['close'].min():.5f} - {df['close'].max():.5f}")
    logger.info(f"Average volume: {df['volume'].mean():,.0f}")
    
    # Test full feature engineering pipeline
    config = PipelineConfig(
        use_forex_optimized=True,
        min_periods=100,
        feature_selection_enabled=True,
        include_metadata=True
    )
    
    pipeline = FeaturePipeline(config)
    features = pipeline.transform(df)
    
    logger.info(f"Generated features: {features.shape}")
    logger.info(f"Feature columns: {list(features.columns)[:10]}...")  # Show first 10
    
    # Check feature quality
    nan_pct = features.isnull().mean()
    logger.info(f"Features with <10% NaN: {(nan_pct < 0.1).sum()}/{len(nan_pct)}")
    
    # Get metadata
    metadata = pipeline.get_metadata()
    logger.info(f"Generated {metadata['feature_stats']['total_features']} total features")
    logger.info(f"Indicators used: {len(metadata['indicators'])}")
    
    logger.info("Realistic sample test completed successfully!")


def main():
    """Run all feature engineering tests."""
    logger = get_logger("main")
    logger.info("="*60)
    logger.info("PHASE 3: FEATURE ENGINEERING FRAMEWORK TESTS")
    logger.info("="*60)
    
    try:
        # Run all tests
        test_individual_indicators()
        logger.info("-" * 40)
        
        test_feature_factory()
        logger.info("-" * 40)
        
        test_feature_calculator()
        logger.info("-" * 40)
        
        test_feature_pipeline()
        logger.info("-" * 40)
        
        test_convenience_function()
        logger.info("-" * 40)
        
        test_data_validation()
        logger.info("-" * 40)
        
        test_with_real_sample()
        logger.info("-" * 40)
        
        logger.info("🎉 ALL PHASE 3 TESTS PASSED SUCCESSFULLY! 🎉")
        logger.info("Feature Engineering Framework is ready for use.")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 