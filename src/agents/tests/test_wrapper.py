#!/usr/bin/env python3
"""Unit tests for MLflow wrapper."""
import pytest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

from src.agents.wrapper import Sb3ModelWrapper


@pytest.mark.unit
class TestSb3ModelWrapper:
    """Unit tests for Sb3ModelWrapper."""

    @pytest.fixture
    def wrapper(self):
        """Create wrapper instance for testing."""
        return Sb3ModelWrapper("ppo")

    @pytest.fixture
    def mock_context(self):
        """Mock MLflow context."""
        context = Mock()
        context.artifacts = {"model_path": "/path/to/model.zip"}
        return context

    @pytest.fixture
    def sample_input(self):
        """Sample input DataFrame for prediction."""
        return pd.DataFrame({
            'close': [1.1234, 1.1235],
            'volume': [1000, 1100],
            'high': [1.1240, 1.1241],
            'low': [1.1230, 1.1231]
        })

    def test_wrapper_initialization(self, wrapper):
        """Test wrapper initialization."""
        assert wrapper._policy_name == "ppo"
        assert wrapper._model is None

    @patch('src.agents.wrapper.stable_baselines3')
    def test_load_context_success(self, mock_sb3, wrapper, mock_context):
        """Test successful context loading."""
        # Setup mock SB3 module
        mock_policy_cls = Mock()
        mock_model = Mock()
        mock_policy_cls.load.return_value = mock_model
        mock_sb3.PPO = mock_policy_cls
        
        # Load context
        wrapper.load_context(mock_context)
        
        # Verify model loading
        mock_policy_cls.load.assert_called_once_with("/path/to/model.zip")
        assert wrapper._model == mock_model

    @patch('src.agents.wrapper.stable_baselines3')
    def test_load_context_case_insensitive(self, mock_sb3, mock_context):
        """Test that policy name is converted to uppercase."""
        # Test lowercase policy name
        wrapper = Sb3ModelWrapper("ppo")
        
        mock_policy_cls = Mock()
        mock_sb3.PPO = mock_policy_cls
        
        wrapper.load_context(mock_context)
        
        # Should access PPO (uppercase) regardless of input case
        assert hasattr(mock_sb3, 'PPO')

    @patch('src.agents.wrapper.stable_baselines3')
    def test_load_context_unknown_algorithm(self, mock_sb3, wrapper, mock_context):
        """Test loading unknown algorithm raises AttributeError."""
        # Remove PPO from mock module
        del mock_sb3.PPO
        
        with pytest.raises(AttributeError, match="SB3 algorithm 'ppo' not found"):
            wrapper.load_context(mock_context)

    @patch('src.agents.wrapper.build_observation')
    def test_predict_success(self, mock_build_obs, wrapper, mock_context, sample_input):
        """Test successful prediction."""
        # Setup loaded model
        mock_model = Mock()
        mock_model.predict.return_value = (np.array([1, 2]), None)
        wrapper._model = mock_model
        
        # Setup observation building
        mock_obs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mock_build_obs.return_value = mock_obs
        
        # Make prediction
        result = wrapper.predict(mock_context, sample_input)
        
        # Verify observation building and prediction
        mock_build_obs.assert_called_once_with(sample_input)
        mock_model.predict.assert_called_once_with(mock_obs, deterministic=True)
        np.testing.assert_array_equal(result, [1, 2])

    def test_predict_without_loaded_model(self, wrapper, mock_context, sample_input):
        """Test prediction fails when model not loaded."""
        # Model not loaded (still None)
        with pytest.raises(RuntimeError, match="Model not loaded"):
            wrapper.predict(mock_context, sample_input)

    @patch('src.agents.wrapper.build_observation')
    def test_predict_single_action(self, mock_build_obs, wrapper, mock_context):
        """Test prediction with single row input."""
        # Single row input
        single_input = pd.DataFrame({'close': [1.1234], 'volume': [1000]})
        
        # Setup model and observation
        mock_model = Mock()
        mock_model.predict.return_value = (1, None)  # Single action
        wrapper._model = mock_model
        
        mock_obs = np.array([1.0, 2.0, 3.0])  # Single observation
        mock_build_obs.return_value = mock_obs
        
        # Make prediction
        result = wrapper.predict(mock_context, single_input)
        
        # Verify result
        assert result == 1

    @patch('src.agents.wrapper.build_observation')
    def test_predict_deterministic_mode(self, mock_build_obs, wrapper, mock_context, sample_input):
        """Test that predictions are made in deterministic mode."""
        # Setup model
        mock_model = Mock()
        mock_model.predict.return_value = (np.array([0]), None)
        wrapper._model = mock_model
        
        # Setup observation
        mock_obs = np.array([[1.0, 2.0]])
        mock_build_obs.return_value = mock_obs
        
        # Make prediction
        wrapper.predict(mock_context, sample_input)
        
        # Verify deterministic=True was used
        mock_model.predict.assert_called_once_with(mock_obs, deterministic=True)

    @patch('src.agents.wrapper.stable_baselines3')
    def test_different_algorithms(self, mock_sb3, mock_context):
        """Test wrapper works with different SB3 algorithms."""
        algorithms = ["ppo", "a2c", "sac", "dqn"]
        
        for algo in algorithms:
            # Setup mock algorithm class
            mock_policy_cls = Mock()
            setattr(mock_sb3, algo.upper(), mock_policy_cls)
            
            # Create wrapper and load context
            wrapper = Sb3ModelWrapper(algo)
            wrapper.load_context(mock_context)
            
            # Verify correct algorithm class was accessed
            getattr(mock_sb3, algo.upper()).load.assert_called_once()

    @patch('src.agents.wrapper.build_observation')
    def test_observation_building_integration(self, mock_build_obs, wrapper, mock_context):
        """Test integration with build_observation function."""
        # Setup model
        mock_model = Mock()
        mock_model.predict.return_value = (np.array([2]), None)
        wrapper._model = mock_model
        
        # Create realistic input data
        input_data = pd.DataFrame({
            'close': [1.1234],
            'volume': [1000],
            'rsi': [65.0],
            'timestamp': pd.to_datetime(['2024-01-01'])
        })
        
        # Setup realistic observation
        mock_obs = np.random.random((1, 16))  # Realistic observation size
        mock_build_obs.return_value = mock_obs
        
        # Make prediction
        result = wrapper.predict(mock_context, input_data)
        
        # Verify build_observation was called with input data
        mock_build_obs.assert_called_once_with(input_data)
        assert result[0] == 2

    def test_wrapper_inheritance(self, wrapper):
        """Test that wrapper properly inherits from MLflow PythonModel."""
        import mlflow.pyfunc
        assert isinstance(wrapper, mlflow.pyfunc.PythonModel)

    @patch('src.agents.wrapper.stable_baselines3')
    def test_error_handling_during_load(self, mock_sb3, wrapper, mock_context):
        """Test error handling when model loading fails."""
        # Setup algorithm class that raises error during load
        mock_policy_cls = Mock()
        mock_policy_cls.load.side_effect = Exception("Model loading failed")
        mock_sb3.PPO = mock_policy_cls
        
        # Loading should propagate the exception
        with pytest.raises(Exception, match="Model loading failed"):
            wrapper.load_context(mock_context) 