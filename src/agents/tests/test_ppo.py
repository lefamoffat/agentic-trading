#!/usr/bin/env python3
"""Unit tests for PPOAgent implementation."""
import pytest
from unittest.mock import Mock, patch
import numpy as np

from src.agents.ppo import PPOAgent
from src.environment import TradingEnv, TradingEnvironmentConfig, FeeStructure

@pytest.mark.unit
class TestPPOAgent:
    """Unit tests for PPOAgent."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock trading environment."""
        config = TradingEnvironmentConfig(
            initial_balance=10000.0,
            fee_structure=FeeStructure.SPREAD_BASED,
        )
        return Mock(spec=TradingEnv)

    @pytest.fixture
    def mock_config_loader(self):
        """Mock configuration loader."""
        loader = Mock()
        loader.reload_config.return_value = {
            "ppo": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64
            }
        }
        return loader

    @patch('src.agents.ppo.PPO')
    @patch('src.agents.ppo.ConfigLoader')
    def test_ppo_agent_initialization(self, mock_config_cls, mock_ppo_cls, mock_env):
        """Test PPOAgent initialization with default config."""
        # Setup mocks
        mock_config_loader = Mock()
        mock_config_loader.reload_config.return_value = {
            "ppo": {"learning_rate": 3e-4, "n_steps": 2048}
        }
        mock_config_cls.return_value = mock_config_loader
        
        mock_model = Mock()
        mock_ppo_cls.return_value = mock_model
        
        # Create agent
        agent = PPOAgent(mock_env)
        
        # Verify initialization
        assert agent.env == mock_env
        assert agent.model == mock_model
        assert agent.hyperparams is None
        
        # Verify PPO was called with correct parameters
        mock_ppo_cls.assert_called_once_with(
            env=mock_env,
            tensorboard_log=None,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048
        )

    @patch('src.agents.ppo.PPO')
    @patch('src.agents.ppo.ConfigLoader')
    def test_ppo_agent_with_hyperparams_override(self, mock_config_cls, mock_ppo_cls, mock_env):
        """Test PPOAgent initialization with hyperparams override."""
        # Setup mocks
        mock_config_loader = Mock()
        mock_config_loader.reload_config.return_value = {
            "ppo": {"learning_rate": 3e-4, "n_steps": 2048}
        }
        mock_config_cls.return_value = mock_config_loader
        
        mock_model = Mock()
        mock_ppo_cls.return_value = mock_model
        
        # Custom hyperparams
        hyperparams = {
            "learning_rate": 1e-3,
            "batch_size": 128
        }
        
        # Create agent with hyperparams
        agent = PPOAgent(mock_env, hyperparams=hyperparams)
        
        # Verify hyperparams were stored and used
        assert agent.hyperparams == hyperparams
        
        # Verify PPO was called with merged config (hyperparams override)
        mock_ppo_cls.assert_called_once_with(
            env=mock_env,
            tensorboard_log=None,
            verbose=1,
            learning_rate=1e-3,  # overridden
            n_steps=2048,       # from config
            batch_size=128      # new parameter
        )

    @patch('src.agents.ppo.PPO')
    @patch('src.agents.ppo.ConfigLoader')
    def test_create_model_method(self, mock_config_cls, mock_ppo_cls, mock_env):
        """Test _create_model method directly."""
        # Setup mocks
        mock_config_loader = Mock()
        mock_config_loader.reload_config.return_value = {
            "ppo": {"learning_rate": 3e-4}
        }
        mock_config_cls.return_value = mock_config_loader
        
        mock_model = Mock()
        mock_ppo_cls.return_value = mock_model
        
        # Create agent and test _create_model
        agent = PPOAgent(mock_env)
        
        # Test creating model with different params
        model_params = {"n_steps": 1024}
        result = agent._create_model(model_params)
        
        assert result == mock_model

    def test_get_model_class(self, mock_env):
        """Test _get_model_class returns PPO class."""
        with patch('src.agents.ppo.PPO') as mock_ppo_cls:
            with patch('src.agents.ppo.ConfigLoader'):
                agent = PPOAgent(mock_env)
                
                model_class = agent._get_model_class()
                assert model_class == mock_ppo_cls

    @patch('src.agents.ppo.PPO')
    @patch('src.agents.ppo.ConfigLoader')
    def test_config_loading_error_handling(self, mock_config_cls, mock_ppo_cls, mock_env):
        """Test handling of missing config sections."""
        # Setup mock with missing ppo section
        mock_config_loader = Mock()
        mock_config_loader.reload_config.return_value = {}  # No ppo section
        mock_config_cls.return_value = mock_config_loader
        
        mock_model = Mock()
        mock_ppo_cls.return_value = mock_model
        
        # Should still work with empty config
        agent = PPOAgent(mock_env)
        
        # Verify PPO was called with minimal parameters
        mock_ppo_cls.assert_called_once_with(
            env=mock_env,
            tensorboard_log=None,
            verbose=1
        )

    @patch('src.agents.ppo.PPO')
    @patch('src.agents.ppo.ConfigLoader')
    def test_tensorboard_disabled(self, mock_config_cls, mock_ppo_cls, mock_env):
        """Test that tensorboard logging is explicitly disabled."""
        # Setup mocks
        mock_config_loader = Mock()
        mock_config_loader.reload_config.return_value = {"ppo": {}}
        mock_config_cls.return_value = mock_config_loader
        
        mock_model = Mock()
        mock_ppo_cls.return_value = mock_model
        
        # Create agent
        agent = PPOAgent(mock_env)
        
        # Verify tensorboard_log=None was passed
        call_args = mock_ppo_cls.call_args
        assert call_args[1]["tensorboard_log"] is None

    @patch('src.agents.ppo.PPO')
    @patch('src.agents.ppo.ConfigLoader')
    def test_verbose_mode_enabled(self, mock_config_cls, mock_ppo_cls, mock_env):
        """Test that verbose mode is enabled by default."""
        # Setup mocks
        mock_config_loader = Mock()
        mock_config_loader.reload_config.return_value = {"ppo": {}}
        mock_config_cls.return_value = mock_config_loader
        
        mock_model = Mock()
        mock_ppo_cls.return_value = mock_model
        
        # Create agent
        agent = PPOAgent(mock_env)
        
        # Verify verbose=1 was passed
        call_args = mock_ppo_cls.call_args
        assert call_args[1]["verbose"] == 1

    @patch('src.agents.ppo.PPO')
    @patch('src.agents.ppo.ConfigLoader')
    def test_inherited_methods_work(self, mock_config_cls, mock_ppo_cls, mock_env):
        """Test that inherited BaseAgent methods work correctly."""
        # Setup mocks
        mock_config_loader = Mock()
        mock_config_loader.reload_config.return_value = {"ppo": {}}
        mock_config_cls.return_value = mock_config_loader
        
        mock_model = Mock()
        mock_model.predict.return_value = (2, None)
        mock_ppo_cls.return_value = mock_model
        
        # Create agent
        agent = PPOAgent(mock_env)
        
        # Test predict method (inherited from BaseAgent)
        obs = np.array([1.0, 2.0, 3.0])
        action = agent.predict(obs)
        
        # Verify prediction works
        assert action == 2
        mock_model.predict.assert_called_once_with(obs, deterministic=True) 