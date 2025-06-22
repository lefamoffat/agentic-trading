#!/usr/bin/env python3
"""Tests for CompositeObservation class."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock

from src.environment.observations.composite import CompositeObservation
from src.environment.config import TradingEnvironmentConfig, FeeStructure, ActionType, RewardSystem
from src.environment.state.position import PositionManager, Position
from src.environment.state.portfolio import PortfolioTracker

@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    return pd.DataFrame({
        'open': [1.0, 1.1, 1.2, 1.3, 1.4],
        'high': [1.05, 1.15, 1.25, 1.35, 1.45],
        'low': [0.95, 1.05, 1.15, 1.25, 1.35],
        'close': [1.02, 1.12, 1.22, 1.32, 1.42],
        'volume': [100, 200, 300, 400, 500]
    })

@pytest.fixture
def position_manager():
    """Create a properly configured mock position manager."""
    manager = Mock(spec=PositionManager)
    manager.is_flat = True
    manager.position = Position.FLAT  # Use actual enum value
    manager.entry_price = None
    manager.current_profit = 0.0
    manager.calculate_unrealized_pnl.return_value = 0.0
    return manager

@pytest.fixture
def portfolio_tracker():
    """Create a properly configured mock portfolio tracker."""
    tracker = Mock(spec=PortfolioTracker)
    tracker.balance = 10000.0
    tracker.initial_balance = 10000.0
    tracker.total_profit = 0.0
    tracker.calculate_position_size.return_value = 1000.0
    return tracker

class TestCompositeObservation:
    """Test cases for CompositeObservation class."""

    @pytest.mark.unit
    def test_init_with_default_parameters(self):
        """Test CompositeObservation initialization with default parameters."""
        config = TradingEnvironmentConfig(
            observation_features=['close', 'volume'],
            include_time_features=True,
            include_portfolio_state=True,
            include_position_state=True
        )
        obs = CompositeObservation(config=config)
        
        assert obs.config == config
        assert obs.market_obs is not None
        assert obs.portfolio_obs is not None
        assert obs.time_obs is not None

    @pytest.mark.unit
    def test_init_with_custom_market_features(self):
        """Test CompositeObservation initialization with custom market features."""
        config = TradingEnvironmentConfig(
            observation_features=['close', 'volume', 'high', 'low'],
            include_time_features=True,
            include_portfolio_state=True,
            include_position_state=True
        )
        obs = CompositeObservation(config=config)
        
        assert obs.config.observation_features == ['close', 'volume', 'high', 'low']

    @pytest.mark.unit
    def test_init_without_time_features(self):
        """Test CompositeObservation initialization without time features."""
        config = TradingEnvironmentConfig(
            observation_features=['close'],
            include_time_features=False,
            include_portfolio_state=True,
            include_position_state=True
        )
        obs = CompositeObservation(config=config)
        
        assert obs.time_obs is None

    @pytest.mark.unit
    def test_observation_size_with_all_features(self):
        """Test observation size with all features enabled."""
        config = TradingEnvironmentConfig(
            observation_features=['open', 'high', 'low', 'close', 'volume'],
            include_time_features=True,
            include_portfolio_state=True,
            include_position_state=True
        )
        obs = CompositeObservation(config=config)
        
        # Market: 5 features + Portfolio: 4 features + Time: 9 features = 18 total
        expected_size = 5 + 4 + 9
        assert obs.observation_size == expected_size

    @pytest.mark.unit
    def test_observation_size_without_time_features(self):
        """Test observation size without time features."""
        config = TradingEnvironmentConfig(
            observation_features=['close', 'volume'],
            include_time_features=False,
            include_portfolio_state=True,
            include_position_state=True
        )
        obs = CompositeObservation(config=config)
        
        # Market: 2 features + Portfolio: 4 features = 6 total
        expected_size = 2 + 4
        assert obs.observation_size == expected_size

    @pytest.mark.unit
    def test_observation_size_custom_features(self):
        """Test observation size with custom feature configuration."""
        config = TradingEnvironmentConfig(
            observation_features=['close'],
            include_time_features=True,
            include_portfolio_state=False,
            include_position_state=False
        )
        obs = CompositeObservation(config=config)
        
        # Market: 1 feature + Time: 9 features = 10 total
        expected_size = 1 + 9
        assert obs.observation_size == expected_size

    @pytest.mark.unit
    def test_get_feature_names_all_features(self):
        """Test feature names with all features enabled."""
        config = TradingEnvironmentConfig(
            observation_features=['close', 'volume'],
            include_time_features=True,
            include_portfolio_state=True,
            include_position_state=True
        )
        obs = CompositeObservation(config=config)
        
        feature_names = obs.get_feature_names()
        
        # Should contain market, portfolio, and time feature names
        assert len(feature_names) > 0
        assert any('close' in name for name in feature_names)
        assert any('volume' in name for name in feature_names)
        assert any('balance' in name for name in feature_names)
        assert any('market_open' in name for name in feature_names)

    @pytest.mark.unit
    def test_get_feature_names_without_time(self):
        """Test feature names without time features."""
        config = TradingEnvironmentConfig(
            observation_features=['close'],
            include_time_features=False,
            include_portfolio_state=True,
            include_position_state=True
        )
        obs = CompositeObservation(config=config)
        
        feature_names = obs.get_feature_names()
        
        # Should not contain time features
        assert not any('market_open' in name for name in feature_names)
        assert not any('time_of_day' in name for name in feature_names)
        
        # Should contain market and portfolio features
        assert any('close' in name for name in feature_names)
        assert any('balance' in name for name in feature_names)

    @pytest.mark.unit
    def test_get_observation_with_all_features(self, sample_data, position_manager, portfolio_tracker):
        """Test observation generation with all features enabled."""
        config = TradingEnvironmentConfig(
            observation_features=['close', 'volume'],
            include_time_features=True,
            include_portfolio_state=True,
            include_position_state=True
        )
        obs = CompositeObservation(config=config)
        
        observation = obs.get_observation(sample_data, 2, position_manager, portfolio_tracker)
        
        assert isinstance(observation, np.ndarray)
        assert observation.dtype == np.float32
        assert len(observation) == obs.observation_size
        assert not np.isnan(observation).any()

    @pytest.mark.unit
    def test_get_observation_without_time_features(self, sample_data, position_manager, portfolio_tracker):
        """Test observation generation without time features."""
        config = TradingEnvironmentConfig(
            observation_features=['close'],
            include_time_features=False,
            include_portfolio_state=True,
            include_position_state=True
        )
        obs = CompositeObservation(config=config)
        
        observation = obs.get_observation(sample_data, 1, position_manager, portfolio_tracker)
        
        assert isinstance(observation, np.ndarray)
        assert observation.dtype == np.float32
        assert len(observation) == obs.observation_size

    @pytest.mark.unit
    def test_get_observation_different_steps(self, sample_data, position_manager, portfolio_tracker):
        """Test observation generation for different time steps."""
        config = TradingEnvironmentConfig(
            observation_features=['close'],
            include_time_features=False,
            include_portfolio_state=False,
            include_position_state=False
        )
        obs = CompositeObservation(config=config)
        
        obs1 = obs.get_observation(sample_data, 0, position_manager, portfolio_tracker)
        obs2 = obs.get_observation(sample_data, 2, position_manager, portfolio_tracker)
        
        # Observations should be different for different steps
        assert not np.array_equal(obs1, obs2)

    @pytest.mark.unit
    def test_observation_consistency_across_calls(self, sample_data, position_manager, portfolio_tracker):
        """Test that observation is consistent across multiple calls with same inputs."""
        config = TradingEnvironmentConfig(
            observation_features=['close', 'volume'],
            include_time_features=True,
            include_portfolio_state=True,
            include_position_state=True
        )
        obs = CompositeObservation(config=config)
        
        obs1 = obs.get_observation(sample_data, 1, position_manager, portfolio_tracker)
        obs2 = obs.get_observation(sample_data, 1, position_manager, portfolio_tracker)
        
        # Should be identical for same inputs
        np.testing.assert_array_equal(obs1, obs2)

    @pytest.mark.unit
    def test_observation_changes_with_different_steps(self, sample_data, position_manager, portfolio_tracker):
        """Test that observation changes appropriately with different time steps."""
        config = TradingEnvironmentConfig(
            observation_features=['close'],
            include_time_features=False,
            include_portfolio_state=False,
            include_position_state=False
        )
        obs = CompositeObservation(config=config)
        
        observations = []
        for step in range(3):
            observation = obs.get_observation(sample_data, step, position_manager, portfolio_tracker)
            observations.append(observation)
        
        # Each observation should be different due to different market data
        for i in range(len(observations)):
            for j in range(i + 1, len(observations)):
                assert not np.array_equal(observations[i], observations[j])

    @pytest.mark.unit
    def test_empty_market_features(self, sample_data, position_manager, portfolio_tracker):
        """Test behavior with empty market features list."""
        config = TradingEnvironmentConfig(
            observation_features=[],
            include_time_features=False,
            include_portfolio_state=True,
            include_position_state=True
        )
        obs = CompositeObservation(config=config)
        
        # Should still work with just portfolio features
        observation = obs.get_observation(sample_data, 0, position_manager, portfolio_tracker)
        
        assert isinstance(observation, np.ndarray)
        assert len(observation) == obs.observation_size
        # Should only contain portfolio features (4 features)
        assert obs.observation_size == 4

    @pytest.mark.unit
    def test_observation_dtype_consistency(self, sample_data, position_manager, portfolio_tracker):
        """Test that observation dtype is consistently float32."""
        config = TradingEnvironmentConfig(
            observation_features=['close', 'volume'],
            include_time_features=True,
            include_portfolio_state=True,
            include_position_state=True
        )
        obs = CompositeObservation(config=config)
        
        observation = obs.get_observation(sample_data, 0, position_manager, portfolio_tracker)
        
        assert observation.dtype == np.float32
        # All values should be finite
        assert np.all(np.isfinite(observation))

    @pytest.mark.unit
    def test_time_feature_integration(self, position_manager, portfolio_tracker):
        """Test integration with time features using datetime index."""
        # Create data with datetime index
        timestamps = pd.date_range('2024-01-01 10:00:00', periods=3, freq='1h')
        timestamped_data = pd.DataFrame({
            'close': [1.0, 1.1, 1.2],
            'volume': [100, 200, 300]
        }, index=timestamps)

        config = TradingEnvironmentConfig(
            observation_features=['close'],
            include_time_features=True,
            include_portfolio_state=False,
            include_position_state=False
        )
        obs = CompositeObservation(config=config)
        
        observation = obs.get_observation(timestamped_data, 1, position_manager, portfolio_tracker)
        
        assert isinstance(observation, np.ndarray)
        assert len(observation) == obs.observation_size
        assert observation.dtype == np.float32

    @pytest.mark.unit
    def test_reset_functionality(self):
        """Test that reset method works correctly."""
        config = TradingEnvironmentConfig(
            observation_features=['close', 'volume'],
            include_time_features=True,
            include_portfolio_state=True,
            include_position_state=True
        )
        obs = CompositeObservation(config=config)
        
        # Reset should not raise any errors
        obs.reset()
        
        # Observation should still work after reset
        sample_data = pd.DataFrame({
            'close': [1.0, 1.1],
            'volume': [100, 200]
        })
        
        # Create properly configured mocks
        position_manager = Mock(spec=PositionManager)
        position_manager.is_flat = True
        position_manager.position = Position.FLAT
        position_manager.entry_price = None
        position_manager.calculate_unrealized_pnl.return_value = 0.0
        
        portfolio_tracker = Mock(spec=PortfolioTracker)
        portfolio_tracker.balance = 10000.0
        portfolio_tracker.initial_balance = 10000.0
        portfolio_tracker.calculate_position_size.return_value = 1000.0
        
        observation = obs.get_observation(sample_data, 0, position_manager, portfolio_tracker)
        assert isinstance(observation, np.ndarray) 