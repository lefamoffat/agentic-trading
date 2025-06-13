#!/usr/bin/env python3
"""
A simple reinforcement learning-based trading strategy.
"""
import asyncio
from typing import Dict

from src.agents.base import BaseAgent
from src.brokers.base import BaseBroker
from src.environments.types import Position
from src.strategies.base import BaseStrategy


class RLStrategy(BaseStrategy):
    """
    A trading strategy driven by a reinforcement learning agent.

    This strategy executes a trading loop that queries an RL agent for actions
    based on the latest market data and executes trades through a broker.
    """

    def __init__(
        self,
        agent: BaseAgent,
        broker: BaseBroker,
        trading_symbol: str,
        timeframe: str,
        position_size: int = 10000,
    ):
        """
        Initialize the RL strategy.

        Args:
            agent (BaseAgent): The reinforcement learning agent.
            broker (BaseBroker): The broker for trade execution.
            trading_symbol (str): The symbol to trade (e.g., "EUR/USD").
            timeframe (str): The timeframe for market data (e.g., "1h").
            position_size (int): The size of each position to open.
        """
        super().__init__(agent, broker)
        self.symbol = trading_symbol
        self.timeframe = timeframe
        self.position_size = position_size
        self.current_position: Dict[str, any] = {}

    async def execute(self) -> None:
        """
        Run the main trading loop.
        """
        self.start()
        self.logger.info(
            f"Executing RL strategy for {self.symbol} on {self.timeframe} timeframe."
        )

        while self._is_running:
            try:
                # 1. Get the latest market data
                # In a real scenario, this would be a stream or the latest candle
                # For this example, we'll assume we get the latest single data point
                latest_data = await self.broker.get_live_price(self.symbol)
                
                # The agent expects a DataFrame-like structure for its observation
                # In a full implementation, you would get a full observation
                # from the environment, including all features. This is simplified.
                observation = self.agent.env._get_observation() # This is a placeholder

                # 2. Get action from the agent
                action = self.agent.predict(observation)
                predicted_position = Position(action)
                
                # 3. Get current position from the broker
                self.current_position = await self.broker.get_open_positions(self.symbol)

                # 4. Execute trade based on the agent's action
                await self._execute_trade(predicted_position)

                # Wait for the next candle/tick
                await asyncio.sleep(60)  # Example: wait 60 seconds

            except Exception as e:
                self.logger.error(f"An error occurred in the trading loop: {e}")
                await asyncio.sleep(60) # Wait before retrying

    async def _execute_trade(self, predicted_position: Position) -> None:
        """
        Execute a trade based on the predicted position.
        """
        is_position_open = bool(self.current_position)
        
        # Case 1: Agent wants to be LONG
        if predicted_position == Position.LONG:
            if is_position_open and self.current_position.get("direction") == "SELL":
                self.logger.info("Closing SHORT position and going LONG.")
                await self.broker.close_position(self.current_position["dealId"])
                await self.broker.open_position("BUY", self.symbol, self.position_size)
            elif not is_position_open:
                self.logger.info("Opening LONG position.")
                await self.broker.open_position("BUY", self.symbol, self.position_size)

        # Case 2: Agent wants to be SHORT
        elif predicted_position == Position.SHORT:
            if is_position_open and self.current_position.get("direction") == "BUY":
                self.logger.info("Closing LONG position and going SHORT.")
                await self.broker.close_position(self.current_position["dealId"])
                await self.broker.open_position("SELL", self.symbol, self.position_size)
            elif not is_position_open:
                self.logger.info("Opening SHORT position.")
                await self.broker.open_position("SELL", self.symbol, self.position_size)
                
        # Case 3: Agent wants to be FLAT
        elif predicted_position == Position.FLAT and is_position_open:
            self.logger.info("Closing open position.")
            await self.broker.close_position(self.current_position["dealId"]) 