#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) Agent.
"""
from typing import Any, Dict, Optional, Type
import inspect
import yaml
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from src.agents.base import BaseAgent
from src.environments.base import BaseTradingEnv
from src.utils.config_loader import ConfigLoader


class PPOAgent(BaseAgent):
    """
    A reinforcement learning agent using the PPO algorithm.

    This class wraps the `stable-baselines3` implementation of the PPO
    algorithm, making it compatible with the project's agent interface.
    """

    def __init__(
        self,
        env: BaseTradingEnv,
        hyperparams: Optional[Dict[str, Any]] = None,
        tensorboard_log_path: Optional[str] = None,
    ):
        """
        Initialize the PPO agent.
        """
        super().__init__(env)
        self.hyperparams = hyperparams
        self.model = self._create_model(
            model_params=self.hyperparams,
            tensorboard_log_path=tensorboard_log_path,
        )

    def _create_model(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        tensorboard_log_path: Optional[str] = None,
    ) -> BaseAlgorithm:
        """
        Create the PPO model instance.

        Args:
            model_params (Optional[Dict[str, Any]]): Hyperparameters for the PPO model.
                If provided, they will override the parameters loaded from the config file.
            tensorboard_log_path (Optional[str]): Path to the directory for TensorBoard logs.

        Returns:
            BaseAlgorithm: An instance of the PPO algorithm.
        """
        self.logger.info("Creating PPO model...")
        
        # Load base config
        config_loader = ConfigLoader()
        agent_config = config_loader.reload_config("agent_config").get("ppo", {})

        # If specific params are provided, they override the base config
        if model_params:
            agent_config.update(model_params)
            
        return PPO(
            env=self.env,
            tensorboard_log=tensorboard_log_path,
            verbose=1,
            **agent_config,
        )

    def learn(self, *args, **kwargs):
        """Delegate the learn method to the underlying stable-baselines3 model."""
        self.model.learn(*args, **kwargs)

    def _get_model_class(self) -> Type[BaseAlgorithm]:
        """
        Get the stable-baselines3 model class for PPO.
        """
        return PPO