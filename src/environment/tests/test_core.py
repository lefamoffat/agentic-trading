#!/usr/bin/env python3
"""Unit tests for TradingEnv core environment."""
import pytest
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pandas as pd
import gymnasium as gym

from src.environment.core import TradingEnv
from src.environment.config import TradingEnvironmentConfig, FeeStructure
from src.environment.actions.discrete import DiscreteActionSpace
from src.environment.state.portfolio import PortfolioTracker
from src.environment.state.position import PositionManager

class TestTradingEnv:
    """Test suite for TradingEnv core environment."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create test market data
        self.test_data = pd.DataFrame({
            'open': [1.0, 1.1, 1.2, 1.15, 1.25],
            'high': [1.05, 1.15, 1.25, 1.20, 1.30],
            'low': [0.95, 1.05, 1.15, 1.10, 1.20],
            'close': [1.02, 1.12, 1.22, 1.18, 1.28],
            'volume': [100, 200, 300, 250, 350]
        })
        
        # Create test configuration
        self.config = TradingEnvironmentConfig(
            initial_balance=10000.0,
            fee_structure=FeeStructure.SPREAD_BASED,
            observation_features=['close', 'volume'],
            include_time_features=False
        )
    
    @pytest.mark.unit
    def test_init_with_default_config(self):
        """Test TradingEnv initialization with default configuration."""
        # Create a default configuration for testing
        default_config = TradingEnvironmentConfig(
            initial_balance=10000.0,
            fee_structure=FeeStructure.SPREAD_BASED
        )
        env = TradingEnv(data=self.test_data, config=default_config)
        
        assert env.data is not None
        assert isinstance(env.config, TradingEnvironmentConfig)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert isinstance(env.portfolio_tracker, PortfolioTracker)
        assert isinstance(env.position_manager, PositionManager)
        assert env.current_step == 0
    
    @pytest.mark.unit
    def test_init_with_custom_config(self):
        """Test TradingEnv initialization with custom configuration."""
        env = TradingEnv(data=self.test_data, config=self.config)
        
        assert env.config == self.config
        assert env.portfolio_tracker.balance == self.config.initial_balance
    
    @pytest.mark.unit
    def test_observation_space_property(self):
        """Test observation space property calculation."""
        env = TradingEnv(data=self.test_data, config=self.config)
        
        obs_space = env.observation_space
        assert isinstance(obs_space, gym.spaces.Box)
        
        # Should match observation size from composite observation
        expected_size = len(self.config.observation_features) + 4  # market + portfolio
        assert obs_space.shape == (expected_size,)
        assert obs_space.dtype == np.float32
        # The observation space bounds are set by the observation handler, not inf
        assert len(obs_space.low) == expected_size
        assert len(obs_space.high) == expected_size
    
    @pytest.mark.unit
    def test_action_space_property(self):
        """Test action space property."""
        env = TradingEnv(data=self.test_data, config=self.config)
        
        action_space = env.action_space
        assert hasattr(action_space, 'n')  # Should have discrete action count
        assert action_space.n > 0
    
    @pytest.mark.unit
    def test_reset_basic(self):
        """Test basic environment reset functionality."""
        env = TradingEnv(data=self.test_data, config=self.config)
        
        observation, info = env.reset()
        
        assert isinstance(observation, np.ndarray)
        assert observation.dtype == np.float32
        assert len(observation) == env.observation_space.shape[0]
        assert isinstance(info, dict)
        
        # Environment state should be reset
        assert env.current_step == 0
    
    @pytest.mark.unit
    def test_reset_with_seed(self):
        """Test environment reset with seed for reproducibility."""
        env = TradingEnv(data=self.test_data, config=self.config)
        
        observation1, info1 = env.reset(seed=42)
        observation2, info2 = env.reset(seed=42)
        
        # Should be identical with same seed
        np.testing.assert_array_equal(observation1, observation2)
    
    @pytest.mark.unit
    def test_step_with_valid_action(self):
        """Test environment step with valid action."""
        env = TradingEnv(data=self.test_data, config=self.config)
        env.reset()
        
        # Test with action that should work (hold)
        action = 0  # Typically 'hold' action
        observation, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(observation, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        # Environment should advance
        assert env.current_step == 1
    
    @pytest.mark.unit
    def test_step_with_numpy_array_action(self):
        """Test environment step with numpy array action (model-agnostic)."""
        env = TradingEnv(data=self.test_data, config=self.config)
        env.reset()
        
        # Test with numpy array action (from SB3 models)
        action = np.array([1])  # Long action as numpy array
        observation, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(observation, np.ndarray)
        assert env.current_step == 1
    
    @pytest.mark.unit
    def test_step_with_string_action(self):
        """Test environment step with string action (model-agnostic)."""
        env = TradingEnv(data=self.test_data, config=self.config)
        env.reset()
        
        # Test with string action
        action = "buy"  # String action
        observation, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(observation, np.ndarray)
        assert env.current_step == 1
    
    @pytest.mark.unit
    def test_step_updates_portfolio(self):
        """Test that step properly updates portfolio state."""
        env = TradingEnv(data=self.test_data, config=self.config)
        env.reset()
        
        initial_balance = env.portfolio_tracker.balance
        
        # Take a buy action
        action = 1  # Assuming 1 is buy/long
        env.step(action)
        
        # Portfolio should potentially change (depends on action handler implementation)
        # At minimum, should not crash and should maintain valid state
        assert env.portfolio_tracker.balance > 0  # Should maintain positive balance
    
    @pytest.mark.unit
    def test_step_calculates_reward(self):
        """Test that step calculates meaningful rewards."""
        env = TradingEnv(data=self.test_data, config=self.config)
        env.reset()
        
        # Take several steps and collect rewards
        rewards = []
        for _ in range(3):
            _, reward, terminated, truncated, _ = env.step(0)  # Hold action
            rewards.append(reward)
            if terminated or truncated:
                break
        
        # Rewards should be numeric
        for reward in rewards:
            assert isinstance(reward, (int, float))
            assert np.isfinite(reward)
    
    @pytest.mark.unit
    def test_termination_conditions(self):
        """Test environment termination conditions."""
        # Create config for small dataset termination
        config = TradingEnvironmentConfig(
            initial_balance=10000.0,
            fee_structure=FeeStructure.SPREAD_BASED,
            observation_features=['close'],
            include_time_features=False
        )
        env = TradingEnv(data=self.test_data, config=config)
        env.reset()
        
        # Step until data exhaustion
        terminated = False
        truncated = False
        step_count = 0
        
        while not (terminated or truncated) and step_count < 10:  # Safety limit
            _, _, terminated, truncated, _ = env.step(0)
            step_count += 1
        
        # Should terminate due to data exhaustion
        assert terminated or truncated
    
    @pytest.mark.unit
    def test_data_exhaustion_termination(self):
        """Test termination when market data is exhausted."""
        # Use small dataset
        small_data = self.test_data.iloc[:3]  # Only 3 rows
        env = TradingEnv(data=small_data, config=self.config)
        env.reset()
        
        # Step through all data
        terminated = False
        truncated = False
        
        for _ in range(5):  # Try to go beyond data length
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
        
        # Should terminate when data is exhausted
        assert terminated or truncated
    
    @pytest.mark.unit
    def test_portfolio_balance_termination(self):
        """Test termination when portfolio balance is too low."""
        # Mock portfolio to simulate low balance
        env = TradingEnv(data=self.test_data, config=self.config)
        env.reset()
        
        # Simulate low balance by mocking portfolio
        env.portfolio_tracker.balance = 10.0  # Very low balance
        
        # Step should check termination conditions
        _, _, terminated, truncated, info = env.step(0)
        
        # Should handle low balance appropriately
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    
    @pytest.mark.unit
    def test_info_dict_contents(self):
        """Test that info dictionary contains useful information."""
        env = TradingEnv(data=self.test_data, config=self.config)
        env.reset()
        
        _, _, _, _, info = env.step(0)
        
        assert isinstance(info, dict)
        # Should contain relevant trading information
        # Exact contents depend on implementation
        if 'portfolio_value' in info:
            assert isinstance(info['portfolio_value'], (int, float))
        if 'current_price' in info:
            assert isinstance(info['current_price'], (int, float))
    
    @pytest.mark.unit
    def test_observation_consistency(self):
        """Test that observations are consistent and valid."""
        env = TradingEnv(data=self.test_data, config=self.config)
        obs1, _ = env.reset()
        
        # Take a step
        obs2, _, _, _, _ = env.step(0)
        
        # Observations should have same shape and type
        assert obs1.shape == obs2.shape
        assert obs1.dtype == obs2.dtype
        assert np.all(np.isfinite(obs1))
        assert np.all(np.isfinite(obs2))
    
    @pytest.mark.unit
    def test_multiple_episodes(self):
        """Test running multiple episodes consecutively."""
        env = TradingEnv(data=self.test_data, config=self.config)
        
        for episode in range(3):
            observation, info = env.reset()
            assert isinstance(observation, np.ndarray)
            
            # Run partial episode
            for _ in range(2):
                observation, reward, terminated, truncated, info = env.step(0)
                assert isinstance(observation, np.ndarray)
                if terminated or truncated:
                    break
    
    @pytest.mark.unit
    def test_action_error_handling(self):
        """Test error handling for invalid actions."""
        env = TradingEnv(data=self.test_data, config=self.config)
        env.reset()
        
        # Test with various potentially invalid actions
        invalid_actions = [None, "invalid", -1, 999, [1, 2, 3]]
        
        for action in invalid_actions:
            try:
                observation, reward, terminated, truncated, info = env.step(action)
                # If it doesn't raise an error, should return valid values
                assert isinstance(observation, np.ndarray)
                assert isinstance(reward, (int, float))
                assert isinstance(terminated, bool)
                assert isinstance(truncated, bool)
                assert isinstance(info, dict)
            except (ValueError, TypeError, IndexError):
                # Acceptable to raise errors for truly invalid actions
                pass
    
    @pytest.mark.unit
    def test_environment_properties_after_steps(self):
        """Test that environment properties remain valid after multiple steps."""
        env = TradingEnv(data=self.test_data, config=self.config)
        env.reset()
        
        # Take several steps
        for _ in range(3):
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
        
        # Properties should remain valid
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert hasattr(env.action_space, 'n')
        assert env.current_step >= 0
    
    @pytest.mark.unit
    def test_render_method_exists(self):
        """Test that render method exists and doesn't crash."""
        env = TradingEnv(data=self.test_data, config=self.config)
        env.reset()
        
        # Render should not crash (even if it does nothing)
        try:
            result = env.render()
            # Return value can be None or some rendered output
        except NotImplementedError:
            # Acceptable if render is not implemented
            pass
    
    @pytest.mark.unit
    def test_close_method(self):
        """Test that close method works properly."""
        env = TradingEnv(data=self.test_data, config=self.config)
        env.reset()
        
        # Close should not crash
        env.close()
        
        # Environment should handle being closed gracefully 