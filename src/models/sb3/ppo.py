#!/usr/bin/env python3
"""PPOAgent - SB3 policy wrapper class (migrated)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Type

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from src.environments.base import BaseTradingEnv
from src.utils.config_loader import ConfigLoader

from .base import BaseAgent

__all__ = ["PPOAgent"]


class PPOAgent(BaseAgent):
    """Reinforcement-learning agent using SB3's PPO algorithm."""

    def __init__(
        self,
        env: BaseTradingEnv,
        hyperparams: Optional[Dict[str, Any]] = None,
        tensorboard_log_path: Optional[str] = None,
    ):
        super().__init__(env)
        self.hyperparams = hyperparams
        self.model = self._create_model(hyperparams, tensorboard_log_path)

    # ------------------------------------------------------------------
    # Base hooks
    # ------------------------------------------------------------------

    def _create_model(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        tensorboard_log_path: Optional[str] = None,
    ) -> BaseAlgorithm:
        config_loader = ConfigLoader()
        agent_conf = config_loader.reload_config("agent_config").get("ppo", {})
        if model_params:
            agent_conf.update(model_params)
        return PPO(env=self.env, tensorboard_log=tensorboard_log_path, verbose=1, **agent_conf)

    def _get_model_class(self) -> Type[BaseAlgorithm]:
        return PPO
