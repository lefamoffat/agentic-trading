#!/usr/bin/env python3
"""Unit tests for BaseAgent abstract class."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

from src.agents.base import BaseAgent
from src.environment import TradingEnv, TradingEnvironmentConfig, FeeStructure

class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""
    
    def _create_model(self, model_params=None):
        """Create a mock model for testing."""
        model = Mock()
        model.learn = Mock()
        model.predict = Mock(return_value=(1, None))
        model.save = Mock()
        return model
    
    def _get_model_class(self):
        """Return mock model class."""
        return Mock

@pytest.mark.unit
class TestBaseAgent:
    """Unit tests for BaseAgent abstract class."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock trading environment."""
        config = TradingEnvironmentConfig(
            initial_balance=10000.0,
            fee_structure=FeeStructure.SPREAD_BASED,
        )
        return Mock(spec=TradingEnv)

    @pytest.fixture
    def agent(self, mock_env):
        """Create a concrete agent instance for testing."""
        return ConcreteAgent(mock_env)

    def test_agent_initialization(self, mock_env):
        """Test BaseAgent initialization."""
        agent = ConcreteAgent(mock_env)
        
        assert agent.env == mock_env
        assert agent.model is None
        assert agent.logger is not None

    def test_train_with_model(self, agent):
        """Test training when model is available."""
        # Set up model
        agent.model = agent._create_model()
        
        # Train agent
        agent.train(total_timesteps=1000)
        
        # Verify model.learn was called correctly
        agent.model.learn.assert_called_once_with(
            total_timesteps=1000, 
            callback=None, 
            reset_num_timesteps=False
        )

    def test_train_without_model_raises_error(self, agent):
        """Test training fails when no model is available."""
        agent.model = None
        
        with pytest.raises(ValueError, match="Model must be created"):
            agent.train(total_timesteps=1000)

    def test_predict_with_model(self, agent):
        """Test prediction when model is available."""
        # Set up model
        agent.model = agent._create_model()
        obs = np.array([1.0, 2.0, 3.0])
        
        # Make prediction
        action = agent.predict(obs)
        
        # Verify prediction call and return value
        agent.model.predict.assert_called_once_with(obs, deterministic=True)
        assert action == 1  # Based on mock return value

    def test_predict_without_model_raises_error(self, agent):
        """Test prediction fails when no model is available."""
        agent.model = None
        obs = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="Model has not been trained"):
            agent.predict(obs)

    def test_save_with_model(self, agent, tmp_path):
        """Test saving model to file."""
        # Set up model
        agent.model = agent._create_model()
        save_path = tmp_path / "model.zip"
        
        # Save model
        agent.save(save_path)
        
        # Verify save was called and directory was created
        agent.model.save.assert_called_once_with(save_path)
        assert save_path.parent.exists()

    def test_save_without_model_raises_error(self, agent, tmp_path):
        """Test saving fails when no model is available."""
        agent.model = None
        save_path = tmp_path / "model.zip"
        
        with pytest.raises(ValueError, match="No model to save"):
            agent.save(save_path)

    @patch('src.agents.base.Path.exists')
    def test_load_existing_file(self, mock_exists, agent, tmp_path):
        """Test loading model from existing file."""
        mock_exists.return_value = True
        load_path = tmp_path / "model.zip"
        
        # Mock the model class load method
        mock_model_cls = Mock()
        mock_loaded_model = Mock()
        mock_model_cls.load.return_value = mock_loaded_model
        
        with patch.object(agent, '_get_model_class', return_value=mock_model_cls):
            agent.load(load_path)
        
        # Verify load was called correctly - SB3 expects path without .zip
        expected_load_path = tmp_path / "model"
        mock_model_cls.load.assert_called_once_with(expected_load_path, env=agent.env)
        assert agent.model == mock_loaded_model

    def test_load_nonexistent_file_raises_error(self, agent, tmp_path):
        """Test loading fails when file doesn't exist."""
        load_path = tmp_path / "nonexistent.zip"
        
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            agent.load(load_path)

    def test_abstract_methods_must_be_implemented(self, mock_env):
        """Test that abstract methods must be implemented in subclasses."""
        with pytest.raises(TypeError):
            BaseAgent(mock_env)  # Should fail - can't instantiate abstract class

    def test_train_with_callback(self, agent):
        """Test training with callback parameter."""
        agent.model = agent._create_model()
        mock_callback = Mock()
        
        agent.train(total_timesteps=500, callback=mock_callback)
        
        agent.model.learn.assert_called_once_with(
            total_timesteps=500,
            callback=mock_callback,
            reset_num_timesteps=False
        )

    def test_predict_returns_integer(self, agent):
        """Test that predict always returns an integer."""
        agent.model = agent._create_model()
        # Mock different return types from model.predict
        agent.model.predict.return_value = (np.int64(2), None)
        
        obs = np.array([1.0, 2.0, 3.0])
        action = agent.predict(obs)
        
        assert isinstance(action, int)
        assert action == 2 