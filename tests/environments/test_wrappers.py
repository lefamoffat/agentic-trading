import pytest
import numpy as np
from unittest.mock import MagicMock

from src.environments.wrappers import EvaluationWrapper
from src.environments.trading_env import Trade
from src.environments.types import Position

@pytest.fixture
def mock_env():
    """Create a mock environment that simulates the trading env's behavior."""
    env = MagicMock()
    
    # Mock the reset method to return initial info
    initial_info = {"portfolio_value": 10000.0, "trade_history": []}
    env.reset.return_value = (np.array([1.0]), initial_info)
    
    # Mock the step method to simulate an episode
    step_infos = [
        {"portfolio_value": 10050.0, "trade_history": []},
        {"portfolio_value": 10020.0, "trade_history": []},
        {
            "portfolio_value": 10100.0, 
            "trade_history": [Trade(entry_price=100, exit_price=101, position=Position.LONG, profit=100.0)]
        },
    ]
    
    # obs, reward, terminated, truncated, info
    env.step.side_effect = [
        (np.array([1.1]), 50.0, False, False, step_infos[0]),
        (np.array([1.0]), -30.0, False, False, step_infos[1]),
        (np.array([1.2]), 80.0, True, False, step_infos[2]), # Episode terminates
    ]
    
    return env

@pytest.mark.unit
class TestEvaluationWrapper:
    """Test suite for the EvaluationWrapper."""

    def test_wrapper_collects_data_over_episode(self, mock_env):
        """
        Test that the wrapper correctly collects portfolio values and the
        final trade history over a full episode.
        """
        wrapper = EvaluationWrapper(mock_env)
        
        # Reset the environment
        obs, info = wrapper.reset()
        
        # Check initial state after reset
        assert wrapper.portfolio_values == [10000.0]
        assert wrapper.trade_history == []
        
        # Run the episode to completion
        done = False
        while not done:
            obs, reward, terminated, truncated, info = wrapper.step(0)
            done = terminated or truncated
            
        # Assert that all data was collected correctly
        assert wrapper.portfolio_values == [10000.0, 10050.0, 10020.0, 10100.0]
        assert len(wrapper.trade_history) == 1
        assert isinstance(wrapper.trade_history[0], Trade)
        assert wrapper.trade_history[0].profit == 100.0
        
    def test_wrapper_reset_clears_data(self, mock_env):
        """
        Test that calling reset clears the collected data.
        """
        wrapper = EvaluationWrapper(mock_env)
        wrapper.reset()
        
        # Step once to populate data
        wrapper.step(0)
        assert len(wrapper.portfolio_values) > 1
        
        # Reset again
        wrapper.reset()
        
        # Verify data is cleared (contains only initial value)
        assert wrapper.portfolio_values == [10000.0]
        assert wrapper.trade_history == [] 