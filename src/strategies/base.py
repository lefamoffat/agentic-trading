#!/usr/bin/env python3
"""Base class for trading strategies.
"""
from abc import ABC, abstractmethod

from src.agents.base import BaseAgent
from src.brokers.base import BaseBroker
from src.utils.logger import get_logger


class BaseStrategy(ABC):
    """Abstract base class for a trading strategy.

    This class defines the common interface for all trading strategies,
    connecting an RL agent's decisions to a broker's execution system.

    Attributes:
        agent (BaseAgent): The RL agent providing trading signals.
        broker (BaseBroker): The broker instance for executing trades.
        logger: The logger for the strategy.

    """

    def __init__(self, agent: BaseAgent, broker: BaseBroker):
        """Initialize the strategy.

        Args:
            agent (BaseAgent): The reinforcement learning agent.
            broker (BaseBroker): The broker for trade execution.

        """
        self.agent = agent
        self.broker = broker
        self.logger = get_logger(self.__class__.__name__)
        self._is_running = False

    @abstractmethod
    async def execute(self) -> None:
        """Run the main trading loop.

        This method should contain the logic for fetching market data,
        getting actions from the agent, and placing orders with the broker.
        """
        raise NotImplementedError

    def start(self) -> None:
        """Start the trading strategy."""
        self.logger.info(f"Starting strategy: {self.__class__.__name__}")
        self._is_running = True

    def stop(self) -> None:
        """Stop the trading strategy."""
        self.logger.info(f"Stopping strategy: {self.__class__.__name__}")
        self._is_running = False
