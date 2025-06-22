#!/usr/bin/env python3
"""Unit tests for agents helpers module."""
import pytest
import numpy as np
import pandas as pd

from src.agents.helpers import build_observation

@pytest.mark.unit
class TestBuildObservation:
    """Unit tests for build_observation function."""

    def test_build_observation_with_market_features(self):
        """Test building observation with market features."""
        # Create sample market data
        data = pd.DataFrame({
            'close': [1.1234, 1.1235],
            'volume': [1000, 1100], 
            'high': [1.1240, 1.1241],
            'low': [1.1230, 1.1231]
        })
        
        obs = build_observation(data)
        
        # Should have market features (4) + portfolio features (4) + time features (9) = 17
        assert obs.shape == (2, 17)
        assert obs.dtype == np.float32
        
        # Check market features (first 4 columns) - use almost_equal for float32 precision
        np.testing.assert_array_almost_equal(obs[:, 0], data['close'].values.astype(np.float32))
        np.testing.assert_array_almost_equal(obs[:, 1], data['volume'].values.astype(np.float32))

    def test_build_observation_single_row(self):
        """Test building observation for single row removes batch dimension."""
        data = pd.DataFrame({
            'close': [1.1234],
            'volume': [1000]
        })
        
        obs = build_observation(data)
        
        # Should remove batch dimension for single example
        assert obs.shape == (15,)  # 2 market + 4 portfolio + 9 time
        assert obs.dtype == np.float32

    def test_build_observation_no_market_features(self):
        """Test building observation with no market columns falls back to OHLCV."""
        # Data with only metadata columns
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01']),
            'symbol': ['EUR/USD']
        })
        
        obs = build_observation(data)
        
        # Should have OHLCV fallback (5) + portfolio (4) + time (9) = 18
        assert obs.shape == (18,)
        
        # Market features should be zeros (no OHLCV columns found)
        assert np.all(obs[:5] == 0.0)

    def test_build_observation_ohlcv_fallback_with_some_columns(self):
        """Test OHLCV fallback when only metadata columns exist, with some OHLCV available."""
        # Include some OHLCV columns but also metadata to trigger fallback mode
        data = pd.DataFrame({
            'close': [1.1234],
            'volume': [1000],
            'timestamp': pd.to_datetime(['2024-01-01']),
            'symbol': ['EUR/USD']
        })
        
        obs = build_observation(data)
        
        # Has market columns so no fallback - close and volume are used as market features
        assert obs.shape == (15,)  # 2 market + 4 portfolio + 9 time
        np.testing.assert_array_almost_equal(obs[:2], [1.1234, 1000])

    def test_build_observation_with_ohlcv_fallback(self):
        """Test with OHLCV columns - they're treated as regular market features."""
        data = pd.DataFrame({
            'close': [1.1234],
            'open': [1.1230],
            'high': [1.1240],
            'low': [1.1225],
            'volume': [1000],
            'timestamp': pd.to_datetime(['2024-01-01'])
        })
        
        obs = build_observation(data)
        
        # Market features are in DataFrame column order: close, open, high, low, volume
        # (timestamp is excluded as metadata)
        expected_market = [1.1234, 1.1230, 1.1240, 1.1225, 1000]
        np.testing.assert_array_almost_equal(obs[:5], expected_market)

    def test_build_observation_partial_ohlcv(self):
        """Test OHLCV fallback with missing columns."""
        data = pd.DataFrame({
            'close': [1.1234],
            'volume': [1000]
            # Missing open, high, low
        })
        
        obs = build_observation(data)
        
        # Market features are in DataFrame column order: close, volume
        # Should be [1.1234, 1000] not OHLCV fallback since we have market columns
        expected_market = [1.1234, 1000]
        np.testing.assert_array_almost_equal(obs[:2], expected_market)

    def test_build_observation_excludes_metadata_columns(self):
        """Test that metadata columns are excluded from market features."""
        data = pd.DataFrame({
            'close': [1.1234],
            'volume': [1000],
            'timestamp': pd.to_datetime(['2024-01-01']),
            'symbol': ['EUR/USD'],
            'date': ['2024-01-01']
        })
        
        obs = build_observation(data)
        
        # Should only use close and volume as market features
        assert obs.shape == (15,)  # 2 market + 4 portfolio + 9 time
        np.testing.assert_array_almost_equal(obs[:2], [1.1234, 1000])

    def test_build_observation_portfolio_features(self):
        """Test that portfolio features are correctly set."""
        data = pd.DataFrame({'close': [1.1234]})
        
        obs = build_observation(data)
        
        # Portfolio features start at index len(market_features)
        portfolio_start = 1  # 1 market feature (close)
        portfolio_features = obs[portfolio_start:portfolio_start+4]
        
        # Should be [1.0, 1.0, 0.0, 0.0] (balance_norm, position_type, entry_price_norm, unrealized_pnl_pct)
        expected_portfolio = [1.0, 1.0, 0.0, 0.0]
        np.testing.assert_array_equal(portfolio_features, expected_portfolio)

    def test_build_observation_time_features(self):
        """Test that time features are correctly set."""
        data = pd.DataFrame({'close': [1.1234]})
        
        obs = build_observation(data)
        
        # Time features start at index len(market_features) + 4
        time_start = 1 + 4  # 1 market + 4 portfolio
        time_features = obs[time_start:time_start+9]
        
        # Should be [1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
        # (market_open, time_of_day_norm, day_monday, day_tuesday, ..., day_sunday)
        expected_time = [1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        np.testing.assert_array_equal(time_features, expected_time)

    def test_build_observation_multiple_rows_batch_dimension(self):
        """Test that multiple rows maintain batch dimension."""
        data = pd.DataFrame({
            'close': [1.1234, 1.1235, 1.1236],
            'volume': [1000, 1100, 1200]
        })
        
        obs = build_observation(data)
        
        # Should maintain batch dimension
        assert obs.shape == (3, 15)  # 3 rows, 15 features each
        
        # Each row should have same portfolio and time features
        for i in range(3):
            # Portfolio features should be same for all rows
            np.testing.assert_array_equal(obs[i, 2:6], [1.0, 1.0, 0.0, 0.0])
            # Time features should be same for all rows  
            np.testing.assert_array_equal(obs[i, 6:], [1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_build_observation_dtype_float32(self):
        """Test that observation dtype is float32."""
        data = pd.DataFrame({
            'close': [1.1234],
            'volume': [1000]
        })
        
        obs = build_observation(data)
        
        assert obs.dtype == np.float32

    def test_build_observation_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        data = pd.DataFrame()
        
        obs = build_observation(data)
        
        # Should still have portfolio + time features for 0-length batch
        assert obs.shape == (0, 18)  # 5 OHLCV fallback + 4 portfolio + 9 time

    def test_build_observation_custom_features(self):
        """Test with custom feature columns."""
        data = pd.DataFrame({
            'custom_indicator_1': [0.5],
            'custom_indicator_2': [0.8],
            'rsi': [65.0],
            'macd': [0.02]
        })
        
        obs = build_observation(data)
        
        # Should have 4 custom market features + 4 portfolio + 9 time = 17
        assert obs.shape == (17,)
        
        # Market features should be the custom indicators - use almost_equal for float32 precision
        expected_market = [0.5, 0.8, 65.0, 0.02]
        np.testing.assert_array_almost_equal(obs[:4], expected_market) 