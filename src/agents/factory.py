#!/usr/bin/env python3
"""Factory for creating RL agents."""
from __future__ import annotations

from typing import Any, Dict, Optional, Type

from src.environment import TradingEnv
from src.utils.logger import get_logger

from src.agents.base import BaseAgent
from src.agents.ppo import PPOAgent

__all__ = ["AgentFactory", "agent_factory"]


class AgentFactory:
    """Factory for creating RL agents with clean interfaces."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self._agents: Dict[str, Type[BaseAgent]] = {
            "PPO": PPOAgent,
        }

    def register_agent(self, name: str, agent_cls: Type[BaseAgent]) -> None:
        """Register a new agent class."""
        self.logger.info(f"Registering agent: {name}")
        self._agents[name] = agent_cls

    def create_agent(
        self,
        name: str,
        env: TradingEnv,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> BaseAgent:
        """Create an agent instance."""
        agent_cls = self._agents.get(name.upper())
        if agent_cls is None:
            self.logger.error(f"Agent '{name}' not found.")
            raise ValueError(f"Agent '{name}' not found.")
        return agent_cls(env=env, hyperparams=hyperparams)


# Global factory instance
agent_factory = AgentFactory() 