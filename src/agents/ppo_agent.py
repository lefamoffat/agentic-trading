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

    def _get_model_class(self) -> Type[BaseAlgorithm]:
        """Get the stable-baselines3 PPO model class."""
        return PPO

    def __init__(self, env: BaseTradingEnv, tensorboard_log_path: Optional[str] = None):
        """
        Initialize the PPO agent.

        Args:
            env (BaseTradingEnv): The trading environment.
            tensorboard_log_path (Optional[str]): Path to the directory for TensorBoard logs.
        """
        super().__init__(env)
        self.tensorboard_log_path = tensorboard_log_path
        self.config_loader = ConfigLoader()

    def _load_model_params(self, params_name: str) -> Dict[str, Any]:
        """Load a named set of model parameters from the agent_config.yaml file."""
        try:
            agent_config = self.config_loader.get_agent_config()
            params = agent_config.get(params_name)

            if params is None:
                self.logger.warning(f"'{params_name}' parameter set not found in agent_config.yaml. Using empty params.")
                return {}
            
            self.logger.info(f"Loaded '{params_name}' parameter set from agent_config.yaml.")
            return params
        except FileNotFoundError:
            self.logger.warning(f"agent_config.yaml not found. Using empty params.")
            return {}

    def _create_model(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        model_params_name: str = "ppo",
    ) -> BaseAlgorithm:
        """
        Create the `stable-baselines3` PPO model instance.

        Args:
            model_params (Optional[Dict[str, Any]]): Hyperparameters for the PPO model.
                If provided, they will override the parameters loaded from the config file.
            model_params_name (str): The name of the parameter set to load from the config file (e.g., 'ppo').

        Returns:
            BaseAlgorithm: An instance of the PPO algorithm.
        """
        # Load default params from the YAML file
        default_params = self._load_model_params(model_params_name)

        # Merge provided params with defaults (provided params take precedence)
        user_params = model_params or {}
        merged_params = {**default_params, **user_params}

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
            tensorboard_log=self.tensorboard_log_path,
            **final_params,
        ) 