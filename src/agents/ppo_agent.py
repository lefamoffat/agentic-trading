#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) Agent.
"""
from typing import Any, Dict, Optional, Type
import inspect

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from src.agents.base import BaseAgent
from src.environments.base import BaseTradingEnv


class PPOAgent(BaseAgent):
    """
    A reinforcement learning agent using the PPO algorithm.

    This class wraps the `stable-baselines3` implementation of the PPO
    algorithm, making it compatible with the project's agent interface.
    """

    def _get_model_class(self) -> Type[BaseAlgorithm]:
        """Get the stable-baselines3 PPO model class."""
        return PPO

    def __init__(self, env: BaseTradingEnv):
        """
        Initialize the PPO agent.

        Args:
            env (BaseTradingEnv): The trading environment.
        """
        super().__init__(env)

    def _create_model(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        tensorboard_log_path: Optional[str] = None,
    ) -> BaseAlgorithm:
        """
        Create the `stable-baselines3` PPO model instance.

        Args:
            model_params (Optional[Dict[str, Any]]): Hyperparameters for the PPO model.
            tensorboard_log_path (Optional[str]): Path to the directory for TensorBoard logs.

        Returns:
            BaseAlgorithm: An instance of the PPO algorithm.
        """
        params = model_params or {}
        
        # Default PPO parameters if not provided
        default_params = {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "verbose": 0,
        }
        
        # Merge provided params with defaults
        merged_params = {**default_params, **params}

        # Filter out unexpected arguments to prevent TypeErrors
        ppo_constructor_args = inspect.signature(PPO).parameters.keys()
        final_params = {
            key: value
            for key, value in merged_params.items()
            if key in ppo_constructor_args
        }

        self.logger.info(f"Creating PPO model with params: {final_params}")
        
        return PPO(
            env=self.env,
            tensorboard_log=tensorboard_log_path,
            **final_params,
        ) 