#!/usr/bin/env python3
"""Abstract base class for Stable-Baselines3 agents (moved from src/agents)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Type

import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm

from src.environments.base import BaseTradingEnv
from src.utils.logger import get_logger

__all__ = ["BaseAgent"]


class BaseAgent(ABC):
    """Abstract base class that wraps an SB3 algorithm with common utilities."""

    def __init__(self, env: BaseTradingEnv):
        self.env = env
        self.model: Optional[BaseAlgorithm] = None
        self.logger = get_logger(self.__class__.__name__)

    # ---------------------------------------------------------------------
    # Sub-class hooks
    # ---------------------------------------------------------------------

    @abstractmethod
    def _create_model(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        tensorboard_log_path: Optional[str] = None,
    ) -> BaseAlgorithm:
        """Instantiate and return the concrete SB3 model."""

    @abstractmethod
    def _get_model_class(self) -> Type[BaseAlgorithm]:
        """Return the SB3 class (e.g., `PPO`)."""

    # ---------------------------------------------------------------------
    # Common operations
    # ---------------------------------------------------------------------

    def train(self, total_timesteps: int, callback: Any = "auto", tb_log_name: str | None = "PPO") -> None:
        if self.model is None:
            raise ValueError("Model must be created in the agent's constructor.")
        self.logger.info(f"Starting training for {total_timesteps} timesteps…")
        self.model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=False)
        self.logger.info("Training complete.")

    def predict(self, obs: pd.DataFrame):  # -> Any
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    # Persistence helpers ---------------------------------------------------

    def save(self, file_path: Path) -> None:
        if self.model is None:
            raise ValueError("No model to save.")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(file_path)
        self.logger.info(f"Model saved to {file_path}")

    def load(self, file_path: Path) -> None:
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        self.logger.info(f"Loading model from {file_path}…")
        model_cls = self._get_model_class()
        self.model = model_cls.load(file_path, env=self.env)
        self.logger.info("Model loaded successfully.")
