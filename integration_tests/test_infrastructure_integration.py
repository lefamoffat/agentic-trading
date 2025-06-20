#!/usr/bin/env python3
"""Integration tests for infrastructure components.

These tests verify that the various infrastructure components work together
correctly, including MLflow, data processing, and system initialization.
"""
import os
import tempfile
from pathlib import Path
import pytest
import pandas as pd
import mlflow
from unittest.mock import patch

from src.utils import mlflow as mlflow_utils
from src.utils.config_loader import ConfigLoader
from src.data.processor import DataProcessor
from src.environment.config import TradingEnvironmentConfig, FeeStructure
from src.environment import load_trading_config


@pytest.mark.integration
class TestMLflowIntegration:
    """Test MLflow integration and experiment tracking."""
    
    def test_mlflow_helper_functions_available(self):
        """Test that MLflow helper functions are accessible."""
        # Test that the helper functions exist and are callable
        assert hasattr(mlflow_utils, 'log_params')
        assert hasattr(mlflow_utils, 'log_metrics')
        assert hasattr(mlflow_utils, 'start_experiment_run')
        assert hasattr(mlflow_utils, 'ensure_experiment')
        
        assert callable(mlflow_utils.log_params)
        assert callable(mlflow_utils.log_metrics)
        assert callable(mlflow_utils.start_experiment_run)

    def test_experiment_creation_and_logging(self):
        """Test creating experiments and logging basic metrics."""
        experiment_name = "test_integration_experiment"
        
        # Test experiment creation (works offline)
        exp_id = mlflow_utils.ensure_experiment(experiment_name)
        assert exp_id is not None
        
        # Test logging within a run context
        with mlflow_utils.start_experiment_run(
            run_name="test_integration_run",
            experiment_name=experiment_name
        ) as run:
            assert run is not None
            
            # Test parameter logging
            test_params = {"learning_rate": 0.001, "batch_size": 32}
            mlflow_utils.log_params(test_params)
            
            # Test metrics logging
            test_metrics = {"accuracy": 0.95, "loss": 0.05}
            mlflow_utils.log_metrics(test_metrics)


@pytest.mark.integration
class TestConfigurationIntegration:
    """Test configuration loading and validation."""
    
    def test_config_loader_functionality(self):
        """Test ConfigLoader class functionality."""
        loader = ConfigLoader()
        
        # Test that config loader can access trading config
        config = loader.get_trading_config()
        assert config is not None
        assert isinstance(config, dict)
        
        # Test that key trading parameters are present
        expected_sections = ['instrument', 'risk', 'trading_hours']
        for section in expected_sections:
            assert section in config

    def test_trading_config_loading_direct(self):
        """Test loading trading configuration from YAML using direct function."""
        config_path = Path("configs/trading_config.yaml")
        
        # Test that config file exists and loads
        assert config_path.exists()
        
        config = load_trading_config(config_path)
        assert config is not None
        assert isinstance(config, TradingEnvironmentConfig)
        
        # Test that configuration has expected attributes
        assert hasattr(config, 'initial_balance')
        assert hasattr(config, 'fee_structure')
        assert hasattr(config, 'observation_features')

    def test_trading_environment_config_creation(self):
        """Test creating TradingEnvironmentConfig from parameters."""
        config = TradingEnvironmentConfig(
            initial_balance=10000.0,
            fee_structure=FeeStructure.SPREAD_BASED,
            observation_features=['close', 'volume', 'high', 'low'],
            include_time_features=True
        )
        
        assert config.initial_balance == 10000.0
        assert config.fee_structure == FeeStructure.SPREAD_BASED
        assert 'close' in config.observation_features
        assert config.include_time_features is True


@pytest.mark.integration
class TestDataProcessingIntegration:
    """Test data processing pipeline integration."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        return pd.DataFrame({
            'open': [1.1000 + i * 0.0001 for i in range(100)],
            'high': [1.1010 + i * 0.0001 for i in range(100)],
            'low': [1.0990 + i * 0.0001 for i in range(100)],
            'close': [1.1005 + i * 0.0001 for i in range(100)],
            'volume': [1000 + i * 10 for i in range(100)]
        }, index=dates)

    def test_data_processor_initialization(self, sample_market_data):
        """Test DataProcessor can be initialized and process data."""
        processor = DataProcessor()
        
        # Test that the processor can handle basic data operations
        assert processor is not None
        
        # Test basic data validation
        assert not sample_market_data.empty
        assert all(col in sample_market_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    def test_environment_data_integration(self, sample_market_data):
        """Test that processed data can be used in trading environment."""
        from src.environment import TradingEnv
        
        config = TradingEnvironmentConfig(
            initial_balance=10000.0,
            fee_structure=FeeStructure.SPREAD_BASED,
            observation_features=['close', 'volume']
        )
        
        # Test environment can be created with processed data
        env = TradingEnv(data=sample_market_data, config=config)
        assert env is not None
        
        # Test basic environment operations
        obs, info = env.reset()
        assert obs is not None
        assert len(obs) == 6  # 2 market features + 4 portfolio features
        
        # Test environment step
        obs, reward, terminated, truncated, info = env.step(0)  # No-op action
        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
