#!/usr/bin/env python3
"""Factory for creating reinforcement learning environments.
"""
from typing import Dict, Type

import pandas as pd

from src.environments.base import BaseTradingEnv
from src.environments.trading_env import TradingEnv
from src.utils.logger import get_logger


class EnvironmentFactory:
    """A factory for creating trading environments.

    This factory allows for the creation of different trading environment
    implementations based on a specified name.
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self._environments: Dict[str, Type[BaseTradingEnv]] = {
            "default": TradingEnv,
        }

    def register_environment(self, name: str, env_class: Type[BaseTradingEnv]) -> None:
        """Register a new environment class.

        Args:
            name (str): The name to register the environment under.
            env_class (Type[BaseTradingEnv]): The class of the environment.

        """
        self.logger.info(f"Registering environment: {name}")
        self._environments[name] = env_class

    def create_environment(
        self,
        name: str,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        **kwargs,
    ) -> BaseTradingEnv:
        """Create an instance of a trading environment.

        Args:
            name (str): The name of the environment to create.
            data (pd.DataFrame): The market data for the environment.
            initial_balance (float): The initial balance for the environment.
            **kwargs: Additional keyword arguments for the environment's constructor.

        Returns:
            BaseTradingEnv: An instance of the specified trading environment.

        Raises:
            ValueError: If the specified environment name is not registered.

        """
        self.logger.info(f"Creating environment '{name}'")
        env_class = self._environments.get(name)

        if not env_class:
            self.logger.error(f"Environment '{name}' not found.")
            raise ValueError(f"Environment '{name}' not found.")

        return env_class(
            data=data, initial_balance=initial_balance, **kwargs
        )


# Global instance of the factory
environment_factory = EnvironmentFactory()
