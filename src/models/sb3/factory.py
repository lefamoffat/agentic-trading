#!/usr/bin/env python3
"""Factory for creating SB3-based RL agents."""
from __future__ import annotations

from typing import Any, Dict, Optional, Type

from src.environments.base import BaseTradingEnv
from src.utils.logger import get_logger

from .base import BaseAgent
from .ppo import PPOAgent

__all__ = ["AgentFactory", "agent_factory"]


class AgentFactory:
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self._agents: Dict[str, Type[BaseAgent]] = {
            "PPO": PPOAgent,
        }

    def register_agent(self, name: str, agent_cls: Type[BaseAgent]) -> None:
        self.logger.info(f"Registering agent: {name}")
        self._agents[name] = agent_cls

    def create_agent(
        self,
        name: str,
        env: BaseTradingEnv,
        hyperparams: Optional[Dict[str, Any]] = None,
        tensorboard_log_path: Optional[str] = None,
    ) -> BaseAgent:
        agent_cls = self._agents.get(name)
        if agent_cls is None:
            self.logger.error(f"Agent '{name}' not found.")
            raise ValueError(f"Agent '{name}' not found.")
        return agent_cls(env=env, hyperparams=hyperparams, tensorboard_log_path=tensorboard_log_path)


agent_factory = AgentFactory()
