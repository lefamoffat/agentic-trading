#!/usr/bin/env python3
"""
Base class for reinforcement learning agents.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Type

import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

from src.environments.base import BaseTradingEnv
from src.utils.logger import get_logger


class BaseAgent(ABC):
    """
    Abstract base class for a reinforcement learning agent.

    This class provides a common interface for different RL agents, wrapping
    the functionality of a `stable-baselines3` algorithm. It handles training,
    prediction, and model persistence.

    Attributes:
        env (BaseTradingEnv): The trading environment for the agent.
        model (Optional[BaseAlgorithm]): The underlying `stable-baselines3` model.
        logger: The logger for the agent.
    """

    def __init__(self, env: BaseTradingEnv):
        """
        Initialize the agent.

        Args:
            env (BaseTradingEnv): The trading environment.
        """
        self.env = env
        self.model: Optional[BaseAlgorithm] = None
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def _create_model(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        tensorboard_log_path: Optional[str] = None,
    ) -> BaseAlgorithm:
        """
        Create the `stable-baselines3` model instance.

        This method should be implemented by subclasses to instantiate the
        specific RL algorithm (e.g., PPO, A2C).

        Args:
            model_params (Optional[Dict[str, Any]]): Hyperparameters for the model.
            tensorboard_log_path (Optional[str]): Path to the directory for TensorBoard logs.

        Returns:
            BaseAlgorithm: An instance of a `stable-baselines3` algorithm.
        """
        raise NotImplementedError
    
    @abstractmethod
    def _get_model_class(self) -> Type[BaseAlgorithm]:
        """
        Get the stable-baselines3 model class.

        Returns:
            Type[BaseAlgorithm]: The class of the underlying RL algorithm.
        """
        raise NotImplementedError

    def train(
        self,
        total_timesteps: int,
        callback: Any = "auto",
        model_params: Optional[Dict[str, Any]] = None,
        tb_log_name: Optional[str] = "PPO",
    ) -> None:
        """
        Train the agent's model.

        If a model already exists, it continues training. Otherwise, it creates
        a new model.

        Args:
            total_timesteps (int): The total number of samples (env steps) to train on.
            callback (Any): A callback function for the training process.
            model_params (Optional[Dict[str, Any]]): Hyperparameters for the model.
            tb_log_name (Optional[str]): The name for the TensorBoard log.
        """
        if self.model is None:
            self.logger.info("Creating new model for training...")
            self.model = self._create_model(model_params=model_params)
        else:
            self.logger.info("Continuing training with existing model...")

        self.logger.info(f"Starting training for {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=False,
        )
        self.logger.info("Training complete.")

    def predict(self, obs: pd.DataFrame) -> Any:
        """
        Predict the next action based on the observation.

        Args:
            obs (pd.DataFrame): The current observation from the environment.

        Returns:
            Any: The predicted action from the agent.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")
        
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def save(self, file_path: Path) -> None:
        """
        Save the trained model to a file.

        Args:
            file_path (Path): The path to save the model file.
        """
        if self.model is None:
            raise ValueError("No model to save.")

        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(file_path)
        self.logger.info(f"Model saved to {file_path}")

    def load(self, file_path: Path) -> None:
        """
        Load a trained model from a file.

        Args:
            file_path (Path): The path to the model file.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")

        self.logger.info(f"Loading model from {file_path}...")
        model_class = self._get_model_class()
        self.model = model_class.load(file_path, env=self.env)
        self.logger.info("Model loaded successfully.") 