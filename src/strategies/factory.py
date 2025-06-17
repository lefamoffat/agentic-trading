#!/usr/bin/env python3
"""Factory for creating trading strategies.
"""
from typing import Dict, Type

from src.agents.base import BaseAgent
from src.brokers.base import BaseBroker
from src.strategies.base import BaseStrategy
from src.strategies.rl_strategy import RLStrategy
from src.utils.logger import get_logger


class StrategyFactory:
    """A factory for creating trading strategies.

    This factory allows for the creation of different strategy implementations
    based on a specified name.
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self._strategies: Dict[str, Type[BaseStrategy]] = {
            "RL": RLStrategy,
            # Other strategies can be registered here
        }

    def register_strategy(self, name: str, strategy_class: Type[BaseStrategy]) -> None:
        """Register a new strategy class.

        Args:
            name (str): The name to register the strategy under.
            strategy_class (Type[BaseStrategy]): The class of the strategy.

        """
        self.logger.info(f"Registering strategy: {name}")
        self._strategies[name] = strategy_class

    def create_strategy(
        self, name: str, agent: BaseAgent, broker: BaseBroker, **kwargs
    ) -> BaseStrategy:
        """Create an instance of a trading strategy.

        Args:
            name (str): The name of the strategy to create.
            agent (BaseAgent): The RL agent for the strategy.
            broker (BaseBroker): The broker for the strategy.
            **kwargs: Additional keyword arguments for the strategy's constructor.

        Returns:
            BaseStrategy: An instance of the specified strategy.

        Raises:
            ValueError: If the specified strategy name is not registered.

        """
        self.logger.info(f"Creating strategy '{name}'")
        strategy_class = self._strategies.get(name)

        if not strategy_class:
            self.logger.error(f"Strategy '{name}' not found.")
            raise ValueError(f"Strategy '{name}' not found.")

        return strategy_class(agent=agent, broker=broker, **kwargs)


# Global instance of the factory
strategy_factory = StrategyFactory()
