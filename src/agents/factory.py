#!/usr/bin/env python3
"""Factory for creating reinforcement learning agents.
"""
from typing import Dict, Type

from src.agents.base import BaseAgent
from src.agents.ppo_agent import PPOAgent
from src.environments.base import BaseTradingEnv
from src.utils.logger import get_logger


class AgentFactory:
    """A factory for creating reinforcement learning agents.

    This factory allows for the creation of different RL agent implementations
    based on a specified name.
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self._agents: Dict[str, Type[BaseAgent]] = {
            "PPO": PPOAgent,
            # Other agents like A2C, SAC can be registered here
        }

    def register_agent(self, name: str, agent_class: Type[BaseAgent]) -> None:
        """Register a new agent class.

        Args:
            name (str): The name to register the agent under.
            agent_class (Type[BaseAgent]): The class of the agent.

        """
        self.logger.info(f"Registering agent: {name}")
        self._agents[name] = agent_class

    def create_agent(
        self,
        name: str,
        env: BaseTradingEnv,
        hyperparams: Dict = None,
        tensorboard_log_path: str = None,
    ) -> BaseAgent:
        """Create an instance of an RL agent.

        Args:
            name (str): The name of the agent to create.
            env (BaseTradingEnv): The trading environment for the agent.
            hyperparams (Dict, optional): Hyperparameters for the agent.
            tensorboard_log_path (str, optional): Path for TensorBoard logs.

        Returns:
            BaseAgent: An instance of the specified agent.

        Raises:
            ValueError: If the specified agent name is not registered.

        """
        self.logger.info(f"Creating agent '{name}'")
        agent_class = self._agents.get(name)

        if not agent_class:
            self.logger.error(f"Agent '{name}' not found.")
            raise ValueError(f"Agent '{name}' not found.")

        return agent_class(
            env=env,
            hyperparams=hyperparams,
            tensorboard_log_path=tensorboard_log_path,
        )


# Global instance of the factory
agent_factory = AgentFactory()
