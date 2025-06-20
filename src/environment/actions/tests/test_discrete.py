#!/usr/bin/env python3
"""Unit tests for DiscreteActionSpace component."""
import pytest
from unittest.mock import Mock

import numpy as np
import torch

from ..discrete import DiscreteActionSpace, TradingAction
from ..base import BaseActionHandler


class TestDiscreteActionSpace:
    """Test suite for DiscreteActionSpace component."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.action_space = DiscreteActionSpace()
        
        # Create mock portfolio for testing
        self.mock_portfolio = Mock()
        self.mock_portfolio.balance = 10000.0
        self.mock_portfolio.position = Mock()
        self.mock_portfolio.position.type = 0  # No position
        
        # Mock current price
        self.current_price = 1.1250
    
    @pytest.mark.unit
    def test_inheritance_from_base_handler(self):
        """Test that DiscreteActionSpace inherits from BaseActionHandler."""
        assert isinstance(self.action_space, BaseActionHandler)
    
    @pytest.mark.unit
    def test_n_property(self):
        """Test the n property returns correct number of actions."""
        assert self.action_space.n == 3  # HOLD, LONG, SHORT
        assert self.action_space.n == len(TradingAction)
    
    @pytest.mark.unit
    def test_sample_method(self):
        """Test the sample method returns valid action indices."""
        for _ in range(10):
            action = self.action_space.sample()
            # Gymnasium returns numpy.int64, which is fine
            assert isinstance(action, (int, np.integer))
            assert 0 <= action < self.action_space.n
    
    @pytest.mark.unit
    def test_contains_valid_actions(self):
        """Test contains method with valid actions."""
        # Test integer actions
        for i in range(self.action_space.n):
            assert self.action_space.contains(i)
        
        # Test numpy array actions
        assert self.action_space.contains(np.array([0]))
        assert self.action_space.contains(np.array([1]))
        assert self.action_space.contains(np.array([2]))
        
        # Test TradingAction enums
        assert self.action_space.contains(TradingAction.OPEN_LONG)
        assert self.action_space.contains(TradingAction.CLOSE)
        assert self.action_space.contains(TradingAction.OPEN_SHORT)
    
    @pytest.mark.unit
    def test_contains_invalid_actions(self):
        """Test contains method with invalid actions."""
        # Test out of range integers
        assert not self.action_space.contains(-1)
        assert not self.action_space.contains(3)
        assert not self.action_space.contains(100)
        
        # Test invalid types
        assert not self.action_space.contains("invalid")
        assert not self.action_space.contains([1, 2])
        assert not self.action_space.contains(None)
    
    @pytest.mark.unit
    def test_normalize_input_integer_actions(self):
        """Test input normalization with integer actions."""
        # Note: Using validate_action instead of _normalize_input since that's the public API
        assert self.action_space.validate_action(0) == TradingAction.OPEN_LONG
        assert self.action_space.validate_action(1) == TradingAction.CLOSE
        assert self.action_space.validate_action(2) == TradingAction.OPEN_SHORT
    
    @pytest.mark.unit
    def test_normalize_input_numpy_arrays(self):
        """Test input normalization with numpy arrays (SB3 compatibility)."""
        # Single element arrays
        assert self.action_space.validate_action(np.array([0])) == TradingAction.OPEN_LONG
        assert self.action_space.validate_action(np.array([1])) == TradingAction.CLOSE
        assert self.action_space.validate_action(np.array([2])) == TradingAction.OPEN_SHORT
        
        # Scalar arrays
        assert self.action_space.validate_action(np.array(0)) == TradingAction.OPEN_LONG
        assert self.action_space.validate_action(np.array(1)) == TradingAction.CLOSE
        assert self.action_space.validate_action(np.array(2)) == TradingAction.OPEN_SHORT
    
    @pytest.mark.unit
    def test_normalize_input_pytorch_tensors(self):
        """Test input normalization with PyTorch tensors."""
        # Single element tensors
        assert self.action_space.validate_action(torch.tensor([0])) == TradingAction.OPEN_LONG
        assert self.action_space.validate_action(torch.tensor([1])) == TradingAction.CLOSE
        assert self.action_space.validate_action(torch.tensor([2])) == TradingAction.OPEN_SHORT
        
        # Scalar tensors
        assert self.action_space.validate_action(torch.tensor(0)) == TradingAction.OPEN_LONG
        assert self.action_space.validate_action(torch.tensor(1)) == TradingAction.CLOSE
        assert self.action_space.validate_action(torch.tensor(2)) == TradingAction.OPEN_SHORT
    
    @pytest.mark.unit
    def test_normalize_input_string_actions(self):
        """Test input normalization with string actions."""
        # Test various string formats for OPEN_LONG
        long_strings = ["open_long", "OPEN_LONG", "long", "LONG", "buy", "BUY"]
        for string in long_strings:
            assert self.action_space.validate_action(string) == TradingAction.OPEN_LONG
        
        # Test various string formats for CLOSE
        close_strings = ["close", "CLOSE", "close_position", "exit", "EXIT"]
        for string in close_strings:
            assert self.action_space.validate_action(string) == TradingAction.CLOSE
        
        # Test various string formats for OPEN_SHORT
        short_strings = ["open_short", "OPEN_SHORT", "short", "SHORT", "sell", "SELL"]
        for string in short_strings:
            assert self.action_space.validate_action(string) == TradingAction.OPEN_SHORT
    
    @pytest.mark.unit
    def test_normalize_input_trading_action_enums(self):
        """Test input normalization with TradingAction enums."""
        assert self.action_space.validate_action(TradingAction.OPEN_LONG) == TradingAction.OPEN_LONG
        assert self.action_space.validate_action(TradingAction.CLOSE) == TradingAction.CLOSE
        assert self.action_space.validate_action(TradingAction.OPEN_SHORT) == TradingAction.OPEN_SHORT
    
    @pytest.mark.unit
    def test_normalize_input_invalid_actions(self):
        """Test input normalization with invalid actions."""
        invalid_actions = [
            -1, 3, 100,  # Out of range integers
            "invalid", "random_string",  # Invalid strings
            [1, 2], {"action": 1},  # Invalid types
            None, np.array([1, 2]),  # Invalid shapes/None
        ]
        
        for action in invalid_actions:
            with pytest.raises((ValueError, TypeError, IndexError)):
                self.action_space.validate_action(action)
    
    @pytest.mark.unit
    def test_framework_compatibility_numpy(self):
        """Test compatibility with NumPy arrays from SB3."""
        # Simulate SB3 model output
        sb3_actions = [
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int32),
            np.array([2], dtype=np.float32),
            np.int64(1),
            np.int32(0),
        ]
        
        for action in sb3_actions:
            normalized = self.action_space.validate_action(action)
            assert isinstance(normalized, TradingAction)
    
    @pytest.mark.unit
    def test_framework_compatibility_pytorch(self):
        """Test compatibility with PyTorch tensors."""
        pytorch_actions = [
            torch.tensor(0, dtype=torch.long),
            torch.tensor([1], dtype=torch.int),
            torch.tensor([2], dtype=torch.float),
        ]
        
        for action in pytorch_actions:
            normalized = self.action_space.validate_action(action)
            assert isinstance(normalized, TradingAction)
    
    @pytest.mark.unit
    def test_action_consistency(self):
        """Test that same input produces same action consistently."""
        # Group inputs that should produce same action
        hold_inputs = [0, np.array([0]), torch.tensor(0)]
        long_inputs = ["buy", "BUY", "long", "LONG"]
        
        # Test consistency within groups
        hold_actions = [self.action_space.validate_action(inp) for inp in hold_inputs]
        assert all(action == TradingAction.OPEN_LONG for action in hold_actions)
        
        long_actions = [self.action_space.validate_action(inp) for inp in long_inputs]
        assert all(action == TradingAction.OPEN_LONG for action in long_actions)
    
    @pytest.mark.unit
    def test_edge_cases_empty_arrays(self):
        """Test handling of edge cases like empty arrays."""
        edge_cases = [
            np.array([]),  # Empty array
            torch.tensor([]),  # Empty tensor
            "",  # Empty string
        ]
        
        for case in edge_cases:
            with pytest.raises((ValueError, IndexError)):
                self.action_space.validate_action(case)
    
    @pytest.mark.unit
    def test_multidimensional_arrays(self):
        """Test handling of multidimensional arrays."""
        multidim_cases = [
            np.array([[1]]),  # 2D array
            torch.tensor([[1]]),  # 2D tensor
        ]
        
        for case in multidim_cases:
            try:
                normalized = self.action_space.validate_action(case)
                assert isinstance(normalized, TradingAction)
            except (ValueError, IndexError):
                # Acceptable to reject multidimensional inputs
                pass 