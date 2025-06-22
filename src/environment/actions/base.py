#!/usr/bin/env python3
"""Base action handler interface for model-agnostic action processing.

This module defines the contract for action handlers that can accept
inputs from any ML framework or model type.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

from src.environment.config import ActionType

class BaseActionHandler(ABC):
    """Abstract base class for model-agnostic action handlers.
    
    This interface ensures all action handlers can accept inputs from:
    - Stable-Baselines3 (numpy arrays)
    - PyTorch (tensors) 
    - TensorFlow (tensors)
    - LLMs (strings/dicts)
    - Human input (integers/strings)
    - Future frameworks
    """
    
    def __init__(self):
        """Initialize the action handler."""
        self.action_type: ActionType = None
    
    @abstractmethod
    def validate_action(self, action: Any) -> Any:
        """Validate and convert any action input to environment-compatible format.
        
        Args:
            action: Action from any source (numpy array, tensor, string, int, etc.)
            
        Returns:
            Environment-compatible action (typically enum or int)
            
        Raises:
            ValueError: If action cannot be converted or is invalid
        """
        pass
    
    @abstractmethod
    def get_action_info(self, action: Any) -> Dict[str, str]:
        """Get human-readable information about an action.
        
        Args:
            action: The action to describe
            
        Returns:
            Dictionary with action details
        """
        pass
    
    @abstractmethod
    def sample(self) -> Any:
        """Sample a random valid action.
        
        Returns:
            Random valid action in the handler's native format
        """
        pass
    
    @property
    @abstractmethod
    def action_space(self):
        """Get the gymnasium action space for this handler.
        
        Returns:
            Gymnasium action space object
        """
        pass
    
    def _normalize_input(self, action: Any) -> Any:
        """Normalize various input types to a standard format.
        
        This helper method handles common conversions across frameworks.
        
        Args:
            action: Raw action input
            
        Returns:
            Normalized action (typically int or float)
            
        Raises:
            ValueError: If input type is not supported
        """
        # Handle numpy arrays (SB3, general ML)
        if isinstance(action, np.ndarray):
            if action.ndim == 0:  # 0-dimensional array (scalar)
                return action.item()
            elif action.ndim == 1 and len(action) == 1:  # 1D array with single element
                return action[0]
            else:
                raise ValueError(f"Cannot convert numpy array with shape {action.shape} to scalar action")
        
        # Handle numpy scalars
        if isinstance(action, (np.integer, np.floating)):
            return action.item()
        
        # Handle PyTorch tensors (if available)
        if hasattr(action, 'item') and hasattr(action, 'dim'):  # Duck typing for torch.Tensor
            if action.dim() == 0:  # Scalar tensor
                return action.item()
            elif action.dim() == 1 and action.size(0) == 1:  # 1D tensor with single element
                return action[0].item()
            else:
                raise ValueError(f"Cannot convert tensor with shape {action.shape} to scalar action")
        
        # Handle TensorFlow tensors (if available)
        if hasattr(action, 'numpy') and hasattr(action, 'shape'):  # Duck typing for tf.Tensor
            numpy_action = action.numpy()
            return self._normalize_input(numpy_action)  # Recurse with numpy array
        
        # Handle basic Python types
        if isinstance(action, (int, float)):
            return action
        
        # Handle enum types (let subclass handle the specific enum)
        if hasattr(action, 'value') and hasattr(action, 'name'):  # Duck typing for enum
            return action
        
        # Handle string inputs (for LLM integration)
        if isinstance(action, str):
            try:
                return int(action)
            except ValueError:
                # Could be action name like "OPEN_LONG" - let subclass handle
                return action
        
        # If we get here, the type is not supported
        raise ValueError(
            f"Unsupported action type: {type(action)}. "
            f"Supported types: int, float, str, numpy arrays/scalars, "
            f"PyTorch tensors, TensorFlow tensors"
        ) 