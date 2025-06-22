#!/usr/bin/env python3
"""Unit tests for AgentFactory."""
import pytest
from unittest.mock import Mock, patch
import gymnasium as gym
import numpy as np

from src.agents.factory import AgentFactory, agent_factory
from src.agents.base import BaseAgent
from src.agents.ppo import PPOAgent
from src.environment import TradingEnv, TradingEnvironmentConfig, FeeStructure

class MockAgent(BaseAgent):
    """Mock agent class for testing."""
    
    def __init__(self, env, hyperparams=None):
        super().__init__(env)
        self.hyperparams = hyperparams
        self.model = Mock()
    
    def _create_model(self, model_params=None):
        return Mock()
    
    def _get_model_class(self):
        return Mock

@pytest.mark.unit
class TestAgentFactory:
    """Unit tests for AgentFactory."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock trading environment with required gym attributes."""
        mock_env = Mock(spec=TradingEnv)
        # Add required gym attributes for SB3
        mock_env.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
        )
        mock_env.action_space = gym.spaces.Discrete(3)
        return mock_env

    @pytest.fixture
    def factory(self):
        """Create a fresh factory instance for testing."""
        return AgentFactory()

    def test_factory_initialization(self, factory):
        """Test AgentFactory initialization."""
        assert hasattr(factory, 'logger')
        assert hasattr(factory, '_agents')
        assert "PPO" in factory._agents
        assert factory._agents["PPO"] == PPOAgent

    def test_register_agent(self, factory):
        """Test registering a new agent class."""
        factory.register_agent("MOCK", MockAgent)
        
        assert "MOCK" in factory._agents
        assert factory._agents["MOCK"] == MockAgent

    def test_create_ppo_agent(self, factory, mock_env):
        """Test creating PPO agent."""
        # Mock the PPO agent constructor to avoid SB3 initialization
        with patch('src.agents.ppo.PPO') as mock_ppo_cls:
            mock_ppo_model = Mock()
            mock_ppo_cls.return_value = mock_ppo_model
            
            agent = factory.create_agent("PPO", mock_env)
            
            assert isinstance(agent, PPOAgent)
            assert agent.env == mock_env

    def test_create_agent_with_hyperparams(self, factory, mock_env):
        """Test creating agent with hyperparameters."""
        hyperparams = {"learning_rate": 1e-3, "batch_size": 128}
        
        # Mock the PPO agent constructor to avoid SB3 initialization
        with patch('src.agents.ppo.PPO') as mock_ppo_cls:
            mock_ppo_model = Mock()
            mock_ppo_cls.return_value = mock_ppo_model
            
            agent = factory.create_agent("PPO", mock_env, hyperparams=hyperparams)
            
            assert isinstance(agent, PPOAgent)
            assert agent.hyperparams == hyperparams

    def test_create_agent_case_insensitive(self, factory, mock_env):
        """Test creating agent with case-insensitive name."""
        with patch('src.agents.ppo.PPO') as mock_ppo_cls:
            mock_ppo_model = Mock()
            mock_ppo_cls.return_value = mock_ppo_model
            
            # Test lowercase
            agent = factory.create_agent("ppo", mock_env)
            assert isinstance(agent, PPOAgent)
            
            # Test mixed case
            agent = factory.create_agent("Ppo", mock_env)
            assert isinstance(agent, PPOAgent)

    def test_create_unknown_agent_raises_error(self, factory, mock_env):
        """Test creating unknown agent raises ValueError."""
        with pytest.raises(ValueError, match="Agent 'UNKNOWN' not found"):
            factory.create_agent("UNKNOWN", mock_env)

    def test_create_custom_registered_agent(self, factory, mock_env):
        """Test creating a custom registered agent."""
        # Register custom agent
        factory.register_agent("CUSTOM", MockAgent)
        
        # Create agent
        agent = factory.create_agent("CUSTOM", mock_env)
        
        assert isinstance(agent, MockAgent)
        assert agent.env == mock_env

    def test_create_custom_agent_with_hyperparams(self, factory, mock_env):
        """Test creating custom agent with hyperparams."""
        factory.register_agent("CUSTOM", MockAgent)
        hyperparams = {"param1": "value1"}
        
        agent = factory.create_agent("CUSTOM", mock_env, hyperparams=hyperparams)
        
        assert isinstance(agent, MockAgent)
        assert agent.hyperparams == hyperparams

    def test_error_logging_on_unknown_agent(self, factory, mock_env):
        """Test that error is logged when agent not found."""
        with patch.object(factory.logger, 'error') as mock_error:
            with pytest.raises(ValueError):
                factory.create_agent("UNKNOWN", mock_env)
            
            mock_error.assert_called_once_with("Agent 'UNKNOWN' not found.")

    def test_register_logging(self, factory):
        """Test that registration is logged."""
        with patch.object(factory.logger, 'info') as mock_info:
            factory.register_agent("TEST", MockAgent)
            
            mock_info.assert_called_once_with("Registering agent: TEST")

    def test_overwrite_existing_agent(self, factory):
        """Test overwriting an existing agent registration."""
        original_ppo = factory._agents["PPO"]
        
        # Overwrite PPO with custom agent
        factory.register_agent("PPO", MockAgent)
        
        assert factory._agents["PPO"] == MockAgent
        assert factory._agents["PPO"] != original_ppo

    def test_multiple_registrations(self, factory):
        """Test registering multiple agents."""
        factory.register_agent("AGENT1", MockAgent)
        factory.register_agent("AGENT2", MockAgent)
        
        assert "AGENT1" in factory._agents
        assert "AGENT2" in factory._agents
        assert factory._agents["AGENT1"] == MockAgent
        assert factory._agents["AGENT2"] == MockAgent

@pytest.mark.unit
class TestGlobalAgentFactory:
    """Unit tests for the global agent_factory instance."""

    def test_global_factory_exists(self):
        """Test that global agent_factory exists."""
        assert agent_factory is not None
        assert isinstance(agent_factory, AgentFactory)

    def test_global_factory_has_ppo(self):
        """Test that global factory has PPO registered."""
        assert "PPO" in agent_factory._agents
        assert agent_factory._agents["PPO"] == PPOAgent

    def test_global_factory_can_create_agents(self):
        """Test that global factory can create agents."""
        mock_env = Mock(spec=TradingEnv)
        # Add required gym attributes for SB3
        mock_env.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
        )
        mock_env.action_space = gym.spaces.Discrete(3)
        
        with patch('src.agents.ppo.PPO') as mock_ppo_cls:
            mock_ppo_model = Mock()
            mock_ppo_cls.return_value = mock_ppo_model
            
            agent = agent_factory.create_agent("PPO", mock_env)
            assert isinstance(agent, PPOAgent) 