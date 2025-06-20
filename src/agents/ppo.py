#!/usr/bin/env python3
"""PPO agent implementation using Stable-Baselines3."""
from __future__ import annotations

from typing import Any, Dict, Optional, Type

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from src.environment import TradingEnv
from src.utils.config_loader import ConfigLoader

from src.agents.base import BaseAgent

__all__ = ["PPOAgent"]


class PPOAgent(BaseAgent):
    """PPO agent implementation with clean configuration management."""

    def __init__(
        self,
        env: TradingEnv,
        hyperparams: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(env)
        self.hyperparams = hyperparams
        self.model = self._create_model(hyperparams)

    def _create_model(
        self,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> BaseAlgorithm:
        """Create PPO model with configuration from agent_config.yaml."""
        config_loader = ConfigLoader()
        agent_conf = config_loader.reload_config("agent_config").get("ppo", {})
        
        # Override with provided hyperparams
        if model_params:
            agent_conf.update(model_params)
            
        return PPO(env=self.env, tensorboard_log=None, verbose=1, **agent_conf)

    def _get_model_class(self) -> Type[BaseAlgorithm]:
        """Return PPO class for loading saved models."""
        return PPO 