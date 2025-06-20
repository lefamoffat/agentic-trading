#!/usr/bin/env python3
"""Discrete action space implementation for trading environment.

This module implements the three-action discrete space: OPEN_LONG, CLOSE, OPEN_SHORT.
"""
from enum import Enum
from typing import Any, Dict, Union

import numpy as np
from gymnasium import spaces

from .base import BaseActionHandler
from ..config import ActionType


class TradingAction(Enum):
    """Enumeration of available trading actions."""
    OPEN_LONG = 0   # Open a long position
    CLOSE = 1       # Close current position (regardless of direction)
    OPEN_SHORT = 2  # Open a short position


class DiscreteActionSpace(BaseActionHandler):
    """Handles discrete trading actions with model-agnostic input support.
    
    This action handler can accept inputs from any ML framework:
    - SB3: numpy arrays
    - PyTorch: tensors
    - TensorFlow: tensors  
    - LLMs: strings or integers
    - Human input: integers or action names
    """
    
    def __init__(self):
        """Initialize the discrete action space."""
        super().__init__()
        self.action_type = ActionType.DISCRETE_THREE
        self._action_space = spaces.Discrete(3)
        self._action_mapping = {
            0: TradingAction.OPEN_LONG,
            1: TradingAction.CLOSE,
            2: TradingAction.OPEN_SHORT,
        }
        self._reverse_mapping = {v: k for k, v in self._action_mapping.items()}
        
        # String mappings for LLM integration
        self._string_mappings = {
            "open_long": TradingAction.OPEN_LONG,
            "long": TradingAction.OPEN_LONG,
            "buy": TradingAction.OPEN_LONG,
            "close": TradingAction.CLOSE,
            "close_position": TradingAction.CLOSE,
            "exit": TradingAction.CLOSE,
            "open_short": TradingAction.OPEN_SHORT,
            "short": TradingAction.OPEN_SHORT,
            "sell": TradingAction.OPEN_SHORT,
        }
    
    def validate_action(self, action: Any) -> TradingAction:
        """Validate and convert any action input to TradingAction enum.
        
        Accepts inputs from any ML framework or source:
        - SB3: numpy.ndarray -> extract scalar -> map to enum
        - PyTorch: torch.Tensor -> extract scalar -> map to enum  
        - TensorFlow: tf.Tensor -> extract scalar -> map to enum
        - LLMs: str -> parse to action name -> map to enum
        - Human: int or str -> map to enum
        
        Args:
            action: Action from any source
            
        Returns:
            TradingAction enum value
            
        Raises:
            ValueError: If action is invalid or cannot be converted
        """
        try:
            # First normalize the input using the base class method
            normalized_action = self._normalize_input(action)
            
            # Handle integer actions (most common case)
            if isinstance(normalized_action, (int, float, np.integer, np.floating)):
                action_int = int(normalized_action)
                if action_int not in self._action_mapping:
                    raise ValueError(f"Invalid action: {action_int}. Must be 0, 1, or 2.")
                return self._action_mapping[action_int]
            
            # Handle string actions (LLM integration)
            elif isinstance(normalized_action, str):
                action_str = normalized_action.lower().strip()
                if action_str in self._string_mappings:
                    return self._string_mappings[action_str]
                else:
                    # Try to parse as action enum name
                    try:
                        return TradingAction[action_str.upper()]
                    except KeyError:
                        raise ValueError(
                            f"Invalid action string: '{action_str}'. "
                            f"Valid options: {list(self._string_mappings.keys())} or {[e.name for e in TradingAction]}"
                        )
            
            # Handle TradingAction enum (already correct format)
            elif isinstance(normalized_action, TradingAction):
                return normalized_action
            
            else:
                raise ValueError(f"Cannot convert normalized action type {type(normalized_action)} to TradingAction")
                
        except Exception as e:
            # Provide helpful error message with original action info
            raise ValueError(
                f"Failed to validate action {action} (type: {type(action)}). "
                f"Error: {str(e)}"
            ) from e
    
    def get_action_info(self, action: Any) -> Dict[str, str]:
        """Get human-readable information about an action.
        
        Args:
            action: The action to describe (any supported format)
            
        Returns:
            Dictionary with action details
        """
        try:
            validated_action = self.validate_action(action)
            
            descriptions = {
                TradingAction.OPEN_LONG: "Open Long Position",
                TradingAction.CLOSE: "Close Current Position", 
                TradingAction.OPEN_SHORT: "Open Short Position",
            }
            
            return {
                "action": validated_action.name,
                "value": validated_action.value,
                "description": descriptions[validated_action],
                "original_input": str(action),
                "input_type": str(type(action).__name__)
            }
        except ValueError as e:
            return {
                "action": "INVALID",
                "value": -1,
                "description": "Invalid Action",
                "original_input": str(action),
                "input_type": str(type(action).__name__),
                "error": str(e)
            }
    
    def sample(self) -> int:
        """Sample a random valid action.
        
        Returns:
            Random integer action (0-2)
        """
        return self._action_space.sample()
    
    def contains(self, action: Any) -> bool:
        """Check if action is valid for this action space.
        
        Args:
            action: Action to validate
            
        Returns:
            True if action is valid, False otherwise
        """
        try:
            self.validate_action(action)
            return True
        except (ValueError, TypeError, IndexError):
            return False
    
    @property
    def action_space(self) -> spaces.Discrete:
        """Get the gymnasium action space.
        
        Returns:
            Discrete action space with 3 actions
        """
        return self._action_space
    
    @property
    def n(self) -> int:
        """Number of discrete actions available.
        
        Returns:
            Number of actions (3)
        """
        return self._action_space.n 