#!/usr/bin/env python3
"""
Base classes for reinforcement learning environments.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd


class BaseTradingEnv(gym.Env, ABC):
    """
    Abstract base class for a trading environment.

    This class defines the interface for a trading environment that is compatible
    with the Gymnasium (formerly OpenAI Gym) API. It handles the core logic for
    stepping through market data, processing actions, and calculating rewards.

    Attributes:
        data (pd.DataFrame): The historical market data for the environment.
        initial_balance (float): The starting balance for each episode.
        current_step (int): The current time step in the episode.
        balance (float): The current account balance.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0):
        """
        Initialize the trading environment.

        Args:
            data (pd.DataFrame): DataFrame containing the market data (OHLCV, features).
            initial_balance (float): The initial account balance for the simulation.
        """
        super().__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = self.initial_balance

        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("Data must be a non-empty pandas DataFrame.")

        if "close" not in self.data.columns:
            raise ValueError("Data must contain 'close' column.")

    @property
    @abstractmethod
    def portfolio_value(self) -> float:
        """
        Return the current total portfolio value (balance + unrealized PnL).
        """
        raise NotImplementedError

    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        """
        Get the observation for the current step.

        Returns:
            np.ndarray: The observation array for the agent.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_info(self) -> Dict[str, Any]:
        """
        Get auxiliary information for the current step.

        Returns:
            Dict[str, Any]: A dictionary containing auxiliary diagnostic information.
        """
        raise NotImplementedError

    @abstractmethod
    def _take_action(self, action: Any) -> None:
        """
        Execute a trading action.

        Args:
            action (Any): The action to be taken by the agent.
        """
        raise NotImplementedError

    @abstractmethod
    def _calculate_reward(self) -> float:
        """
        Calculate the reward for the current step.

        Returns:
            float: The reward value.
        """
        raise NotImplementedError

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Run one timestep of the environment's dynamics.

        Args:
            action (Any): An action provided by the agent.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: A tuple containing:
                - observation (object): Agent's observation of the current environment.
                - reward (float): Amount of reward returned after previous action.
                - terminated (bool): Whether the episode has ended.
                - truncated (bool): Whether the episode was truncated.
                - info (dict): Contains auxiliary diagnostic information.
        """
        self._take_action(action)
        self.current_step += 1

        reward = self._calculate_reward()
        terminated = (
            self.portfolio_value <= 0 or self.current_step >= len(self.data) - 1
        )
        truncated = False

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Args:
            seed (Optional[int]): The random seed for the environment.
            options (Optional[Dict[str, Any]]): Additional options for resetting.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: The initial observation and info dictionary.
        """
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.current_step = 0
        return self._get_observation(), self._get_info()

    def render(self, mode: str = "human") -> Any:
        """
        Render the environment.

        Args:
            mode (str): The mode to render with ('human' or 'ansi').
        """
        if mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Balance: {self.balance:.2f}")
            print(f"Info: {self._get_info()}")
        elif mode == "ansi":
            return f"Step: {self.current_step}, Balance: {self.balance:.2f}"

    def close(self) -> None:
        """Perform any necessary cleanup."""
        print("Closing trading environment.") 