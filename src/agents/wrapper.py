#!/usr/bin/env python3
"""SB3 Agent wrapper for Aim ML tracking integration.

This module provides wrapper functionality for SB3 agents to work with
the generic ML tracking system (currently Aim backend).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import stable_baselines3
from pathlib import Path
from typing import Optional, Any, Dict, Union

from src.agents.helpers import build_observation

__all__ = ["Sb3AimWrapper"]

class Sb3AimWrapper:
    """Aim-compatible wrapper for SB3 agents.
    
    This wrapper allows SB3 agents to be used with the Aim ML tracking
    system for experiment tracking and model management.
    """

    def __init__(self, agent_type: str, model_path: Optional[Union[str, Path]] = None):
        """
        Initialize the wrapper.
        
        Args:
            agent_type: SB3 algorithm name (e.g., "PPO", "A2C", "SAC")
            model_path: Path to saved model (optional)
        """
        self.agent_type = agent_type.upper()
        self.model_path = model_path
        self._model = None
        
        # Validate agent type
        if not hasattr(stable_baselines3, self.agent_type):
            raise ValueError(f"Unknown SB3 algorithm: {self.agent_type}")

    def load_model(self, model_path: Union[str, Path], env: Optional[Any] = None) -> None:
        """
        Load SB3 model from file.
        
        Args:
            model_path: Path to the saved model
            env: Optional environment for the model
        """
        try:
            model_cls = getattr(stable_baselines3, self.agent_type)
            self._model = model_cls.load(str(model_path), env=env)
            self.model_path = Path(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.agent_type} model from {model_path}: {e}")

    def predict(self, observation: Union[pd.DataFrame, np.ndarray], deterministic: bool = True) -> np.ndarray:
        """
        Make predictions using the loaded SB3 model.
        
        Args:
            observation: Market observation data
            deterministic: Whether to use deterministic policy
            
        Returns:
            Predicted actions
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Convert DataFrame to observation if needed
        if isinstance(observation, pd.DataFrame):
            obs = build_observation(observation)
        else:
            obs = observation
            
        actions, _ = self._model.predict(obs, deterministic=deterministic)
        return actions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self._model is None:
            return {"loaded": False}
            
        return {
            "loaded": True,
            "algorithm": self.agent_type,
            "model_path": str(self.model_path) if self.model_path else None,
            "policy": str(type(self._model.policy)),
            "observation_space": str(self._model.observation_space),
            "action_space": str(self._model.action_space)
        } 