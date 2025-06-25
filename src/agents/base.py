#!/usr/bin/env python3
"""Abstract base class for reinforcement learning agents."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Type

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from src.environment import TradingEnv
from src.utils.logger import get_logger

__all__ = ["BaseAgent"]

class BaseAgent(ABC):
    """Abstract base class for RL agents with clean interfaces."""

    def __init__(self, env: TradingEnv):
        self.env = env
        self.model: Optional[BaseAlgorithm] = None
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def _create_model(
        self,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> BaseAlgorithm:
        """Create and return the SB3 model instance."""

    @abstractmethod
    def _get_model_class(self) -> Type[BaseAlgorithm]:
        """Return the SB3 algorithm class."""

    def train(self, total_timesteps: int, callback: Any = None) -> None:
        """Train the agent for specified timesteps."""
        if self.model is None:
            raise ValueError("Model must be created in the agent's constructor.")
        self.logger.debug(f"Training with callback id={id(callback)}")
        self.logger.info(f"Starting training for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=False)
        self.logger.info("Training complete.")

    def predict(self, obs: np.ndarray) -> int:
        """Get action prediction from observation."""
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def save(self, file_path: Path) -> None:
        """Save model to specified path.
        
        Note: Stable-Baselines3 automatically saves models as .zip files.
        If the provided path doesn't end with .zip, SB3 will add it automatically.
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        # Create parent directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # SB3 handles .zip extension automatically
        # If path ends with .zip, SB3 uses it as-is
        # If path doesn't end with .zip, SB3 adds .zip
        self.model.save(file_path)
        
        self.logger.info(f"Model saved to {file_path}")

    def load(self, file_path: Path) -> None:
        """Load model from specified path.
        
        Note: Stable-Baselines3 expects the path without .zip extension.
        SB3 will automatically look for the .zip file.
        """
        # For loading, SB3 expects path without .zip extension
        if file_path.suffix == '.zip':
            load_path = file_path.with_suffix('')
            zip_path = file_path
        else:
            load_path = file_path
            zip_path = file_path.with_suffix('.zip')
            
        # Check if the .zip file actually exists
        if not zip_path.exists():
            raise FileNotFoundError(f"Model file not found: {zip_path}")
        
        self.logger.info(f"Loading model from {zip_path}...")
        model_cls = self._get_model_class()
        
        # SB3 load expects the path without .zip extension
        self.model = model_cls.load(load_path, env=self.env)
        self.logger.info("Model loaded successfully.") 