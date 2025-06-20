#!/usr/bin/env python3
"""Unit tests for CompositeObservation component."""
import pytest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from ..composite import CompositeObservation
from ..market import MarketObservation
from ..portfolio import PortfolioObservation
from ..time_features import TimeObservation
from ...state.position import Position


class TestCompositeObservation:
    """Test suite for CompositeObservation component."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create test data
        self.test_data = pd.DataFrame({
            'open': [1.0, 1.1, 1.2],
            'high': [1.05, 1.15, 1.25],
            'low': [0.95, 1.05, 1.15],
            'close': [1.02, 1.12, 1.22],
            'volume': [100, 200, 300]
        })
        
        # Create mock position manager with proper position enum
        self.mock_position_manager = Mock()
        self.mock_position_manager.position = Position.FLAT  # Return actual enum
        self.mock_position_manager.is_flat = True
        self.mock_position_manager.entry_price = None
        self.mock_position_manager.current_position = None
        self.mock_position_manager.get_position_size.return_value = 0.0
        self.mock_position_manager.get_unrealized_pnl.return_value = 0.0
        self.mock_position_manager.calculate_unrealized_pnl.return_value = 0.0
        
        # Create mock portfolio tracker
        self.mock_portfolio_tracker = Mock()
        self.mock_portfolio_tracker.balance = 10000.0
        self.mock_portfolio_tracker.equity = 10000.0
        self.mock_portfolio_tracker.initial_balance = 10000.0
        self.mock_portfolio_tracker.get_total_value.return_value = 10000.0
        self.mock_portfolio_tracker.calculate_position_size.return_value = 1000.0
    
    @pytest.mark.unit
    def test_init_with_default_parameters(self):
        """Test CompositeObservation initialization with default parameters."""
        obs = CompositeObservation(market_features=['close', 'volume'])
        
        assert obs.include_time_features is True
        assert obs.include_portfolio_state is True
        assert obs.include_position_state is True
        assert isinstance(obs.market_obs, MarketObservation)
        assert isinstance(obs.portfolio_obs, PortfolioObservation)
        assert isinstance(obs.time_obs, TimeObservation)
    
    @pytest.mark.unit
    def test_init_with_custom_market_features(self):
        """Test CompositeObservation initialization with custom market features."""
        market_features = ['close', 'volume', 'rsi']
        obs = CompositeObservation(market_features=market_features)
        
        assert obs.market_obs.features == market_features
    
    @pytest.mark.unit
    def test_init_without_time_features(self):
        """Test CompositeObservation initialization without time features."""
        obs = CompositeObservation(market_features=['close'], include_time_features=False)
        
        assert obs.include_time_features is False
        assert obs.time_obs is None
    
    @pytest.mark.unit
    def test_observation_size_with_all_features(self):
        """Test observation size with all features enabled."""
        obs = CompositeObservation(
            market_features=['open', 'high', 'low', 'close', 'volume'],
            include_time_features=True
        )
        
        # Market: 5 features + Portfolio: 4 features + Time: 9 features = 18 total
        expected_size = 5 + 4 + 9
        assert obs.observation_size == expected_size
    
    @pytest.mark.unit
    def test_observation_size_without_time_features(self):
        """Test observation size without time features."""
        obs = CompositeObservation(
            market_features=['close', 'volume'],
            include_time_features=False
        )
        
        # Market: 2 features + Portfolio: 4 features = 6 total
        expected_size = 2 + 4
        assert obs.observation_size == expected_size
    
    @pytest.mark.unit
    def test_observation_size_custom_features(self):
        """Test observation size with custom feature configuration."""
        obs = CompositeObservation(
            market_features=['close'],
            include_time_features=True
        )
        
        # Market: 1 feature + Portfolio: 4 features + Time: 9 features = 14 total
        expected_size = 1 + 4 + 9
        assert obs.observation_size == expected_size
    
    @pytest.mark.unit
    def test_get_feature_names_all_features(self):
        """Test feature names with all features enabled."""
        obs = CompositeObservation(
            market_features=['close', 'volume'],
            include_time_features=True
        )
        
        feature_names = obs.get_feature_names()
        
        # Should contain market + portfolio + time features
        assert 'close' in feature_names
        assert 'volume' in feature_names
        assert 'balance_normalized' in feature_names
        assert 'position_type' in feature_names
        assert 'entry_price_normalized' in feature_names
        assert 'unrealized_pnl_pct' in feature_names
        assert 'market_open' in feature_names
        assert 'time_of_day_normalized' in feature_names
        assert 'day_monday' in feature_names
        
        # Check total count
        assert len(feature_names) == 2 + 4 + 9  # market + portfolio + time
    
    @pytest.mark.unit
    def test_get_feature_names_without_time(self):
        """Test feature names without time features."""
        obs = CompositeObservation(
            market_features=['close'],
            include_time_features=False
        )
        
        feature_names = obs.get_feature_names()
        
        # Should contain only market + portfolio features
        assert 'close' in feature_names
        assert 'balance_normalized' in feature_names
        assert 'market_open' not in feature_names
        
        # Check total count
        assert len(feature_names) == 1 + 4  # market + portfolio
    
    @pytest.mark.unit
    def test_get_observation_with_all_features(self):
        """Test observation generation with all features enabled."""
        obs = CompositeObservation(
            market_features=['close', 'volume'],
            include_time_features=True
        )
        
        observation = obs.get_observation(
            self.test_data, 
            current_step=0,
            position_manager=self.mock_position_manager,
            portfolio_tracker=self.mock_portfolio_tracker
        )
        
        assert isinstance(observation, np.ndarray)
        assert observation.dtype == np.float32
        assert len(observation) == 2 + 4 + 9  # market + portfolio + time
        
        # Check that all values are finite
        assert np.all(np.isfinite(observation))
    
    @pytest.mark.unit
    def test_get_observation_without_time_features(self):
        """Test observation generation without time features."""
        obs = CompositeObservation(
            market_features=['close'],
            include_time_features=False
        )
        
        observation = obs.get_observation(
            self.test_data,
            current_step=0,
            position_manager=self.mock_position_manager,
            portfolio_tracker=self.mock_portfolio_tracker
        )
        
        assert isinstance(observation, np.ndarray)
        assert len(observation) == 1 + 4  # market + portfolio only
        assert np.all(np.isfinite(observation))
    
    @pytest.mark.unit
    def test_get_observation_different_steps(self):
        """Test observation generation for different time steps."""
        obs = CompositeObservation(
            market_features=['close'],
            include_time_features=False
        )
        
        # Test different steps
        for step in range(len(self.test_data)):
            observation = obs.get_observation(
                self.test_data,
                current_step=step,
                position_manager=self.mock_position_manager,
                portfolio_tracker=self.mock_portfolio_tracker
            )
            
            assert isinstance(observation, np.ndarray)
            assert len(observation) == 1 + 4
            assert np.all(np.isfinite(observation))
    
    @pytest.mark.unit
    def test_observation_consistency_across_calls(self):
        """Test that observation is consistent across multiple calls with same inputs."""
        obs = CompositeObservation(
            market_features=['close', 'volume'],
            include_time_features=True
        )
        
        observation1 = obs.get_observation(
            self.test_data,
            current_step=0,
            position_manager=self.mock_position_manager,
            portfolio_tracker=self.mock_portfolio_tracker
        )
        observation2 = obs.get_observation(
            self.test_data,
            current_step=0,
            position_manager=self.mock_position_manager,
            portfolio_tracker=self.mock_portfolio_tracker
        )
        
        np.testing.assert_array_equal(observation1, observation2)
    
    @pytest.mark.unit
    def test_observation_changes_with_different_steps(self):
        """Test that observation changes appropriately with different time steps."""
        obs = CompositeObservation(
            market_features=['close'],
            include_time_features=False
        )
        
        observation1 = obs.get_observation(
            self.test_data,
            current_step=0,
            position_manager=self.mock_position_manager,
            portfolio_tracker=self.mock_portfolio_tracker
        )
        observation2 = obs.get_observation(
            self.test_data,
            current_step=1,
            position_manager=self.mock_position_manager,
            portfolio_tracker=self.mock_portfolio_tracker
        )
        
        # Market features should change (different close price)
        # Portfolio features may be the same if portfolio state unchanged
        assert not np.array_equal(observation1, observation2)
    
    @pytest.mark.unit
    def test_empty_market_features(self):
        """Test behavior with empty market features list."""
        obs = CompositeObservation(
            market_features=[],
            include_time_features=False
        )
        
        # Should only have portfolio features
        assert obs.observation_size == 4  # Only portfolio features
        
        observation = obs.get_observation(
            self.test_data,
            current_step=0,
            position_manager=self.mock_position_manager,
            portfolio_tracker=self.mock_portfolio_tracker
        )
        assert len(observation) == 4
    
    @pytest.mark.unit
    def test_observation_dtype_consistency(self):
        """Test that observation dtype is consistently float32."""
        obs = CompositeObservation(
            market_features=['close', 'volume'],
            include_time_features=True
        )
        
        observation = obs.get_observation(
            self.test_data,
            current_step=0,
            position_manager=self.mock_position_manager,
            portfolio_tracker=self.mock_portfolio_tracker
        )
        assert observation.dtype == np.float32
    
    @pytest.mark.unit
    def test_time_feature_integration(self):
        """Test integration with time features using datetime index."""
        # Create data with datetime index
        timestamps = pd.date_range('2024-01-01 10:00:00', periods=3, freq='1h')
        timestamped_data = pd.DataFrame({
            'close': [1.0, 1.1, 1.2],
            'volume': [100, 200, 300]
        }, index=timestamps)
        
        obs = CompositeObservation(
            market_features=['close'],
            include_time_features=True
        )
        
        observation = obs.get_observation(
            timestamped_data,
            current_step=0,
            position_manager=self.mock_position_manager,
            portfolio_tracker=self.mock_portfolio_tracker
        )
        
        # Should include market + portfolio + time features
        assert len(observation) == 1 + 4 + 9
        assert np.all(np.isfinite(observation))
        
        # Time features should be meaningful (not all zeros)
        time_features = observation[-9:]  # Last 9 features are time features
        assert np.any(time_features > 0)  # Should have some non-zero time features 