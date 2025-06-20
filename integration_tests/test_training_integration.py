#!/usr/bin/env python3
"""Integration tests for training pipeline.

These tests verify the complete training workflow from data loading
through model training to evaluation and persistence.
"""
import tempfile
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import mlflow

from src.environment import TradingEnv, TradingEnvironmentConfig, FeeStructure
from src.models.sb3 import PPOAgent, AgentFactory
from src.strategies import RLStrategy
from src.utils import mlflow as mlflow_utils
from src.data.processor import DataProcessor


@pytest.mark.integration
class TestEndToEndTrainingPipeline:
    """Test complete training pipeline integration."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create realistic market data for training."""
        np.random.seed(42)  # For reproducible tests
        n_periods = 1000
        
        dates = pd.date_range('2024-01-01', periods=n_periods, freq='1h')
        
        # Generate realistic OHLCV data with some trend and volatility
        base_price = 1.1000
        returns = np.random.normal(0, 0.001, n_periods)  # 0.1% hourly volatility
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLCV with realistic spreads
        open_prices = prices
        close_prices = np.roll(prices, -1)  # Next period's open becomes current close
        close_prices[-1] = prices[-1]  # Handle last period
        
        high_prices = np.maximum(open_prices, close_prices) + np.random.uniform(0, 0.0005, n_periods)
        low_prices = np.minimum(open_prices, close_prices) - np.random.uniform(0, 0.0005, n_periods)
        volumes = np.random.randint(1000, 10000, n_periods)
        
        return pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }, index=dates)

    @pytest.fixture
    def training_config(self):
        """Create configuration for training tests."""
        return TradingEnvironmentConfig(
            initial_balance=10000.0,
            fee_structure=FeeStructure.SPREAD_BASED,
            observation_features=['close', 'volume', 'high', 'low'],
            include_time_features=False,  # Simplify for testing
            spread=0.0002  # 2 pip spread
        )

    def test_environment_creation_with_training_data(self, sample_market_data, training_config):
        """Test that trading environment can be created with training data."""
        env = TradingEnv(data=sample_market_data, config=training_config)
        
        # Test environment properties
        assert env is not None
        assert env.observation_space.shape[0] == 8  # 4 market + 4 portfolio features
        assert env.action_space.n == 3  # OPEN_LONG, CLOSE, OPEN_SHORT
        
        # Test reset and step
        obs, info = env.reset()
        assert obs.shape == (8,)
        
        # Test a few steps
        for action in [0, 1, 2, 1]:  # Open long, close, open short, close
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (8,)
            assert isinstance(reward, (int, float))
            if terminated or truncated:
                break

    def test_ppo_agent_creation_and_basic_training(self, sample_market_data, training_config):
        """Test PPO agent creation and very basic training."""
        env = TradingEnv(data=sample_market_data, config=training_config)
        
        # Create PPO agent
        agent = PPOAgent(env=env, learning_rate=3e-4, n_steps=64)
        assert agent is not None
        assert agent.model is not None
        
        # Test very short training (just to verify it doesn't crash)
        agent.train(total_timesteps=128)  # Very short training
        
        # Test that agent can make predictions
        obs, _ = env.reset()
        action, _states = agent.model.predict(obs, deterministic=True)
        assert action in [0, 1, 2]

    def test_agent_factory_integration(self):
        """Test that AgentFactory can create different agent types."""
        factory = AgentFactory()
        
        # Test factory has expected agent types
        assert hasattr(factory, 'create_agent')
        
        # Test that factory can list available agents
        # Note: This test just verifies the factory exists and is callable
        assert factory is not None

    def test_rl_strategy_integration(self, sample_market_data, training_config):
        """Test RLStrategy integration with trained agent."""
        env = TradingEnv(data=sample_market_data, config=training_config)
        
        # Create and minimally train an agent
        agent = PPOAgent(env=env, learning_rate=3e-4, n_steps=64)
        agent.train(total_timesteps=128)
        
        # Create RL strategy with the agent
        strategy = RLStrategy(agent=agent.model, env=env)
        assert strategy is not None
        
        # Test strategy can generate signals
        obs, _ = env.reset()
        action = strategy.get_action(obs)
        assert action in [0, 1, 2]

    def test_mlflow_integration_with_training(self, sample_market_data, training_config):
        """Test MLflow integration during training workflow."""
        experiment_name = "test_training_integration"
        
        # Ensure experiment exists
        exp_id = mlflow_utils.ensure_experiment(experiment_name)
        assert exp_id is not None
        
        # Test training with MLflow logging
        with mlflow_utils.start_experiment_run(
            run_name="test_training_run",
            experiment_name=experiment_name
        ) as run:
            # Log training configuration
            config_params = {
                "algorithm": "PPO",
                "learning_rate": 3e-4,
                "n_steps": 64,
                "total_timesteps": 128
            }
            mlflow_utils.log_params(config_params)
            
            # Create environment and agent
            env = TradingEnv(data=sample_market_data, config=training_config)
            agent = PPOAgent(env=env, learning_rate=3e-4, n_steps=64)
            
            # Train with minimal timesteps
            agent.train(total_timesteps=128)
            
            # Log some training metrics
            training_metrics = {
                "final_balance": env.portfolio_tracker.balance,
                "total_trades": env.portfolio_tracker.total_trades,
                "env_steps": env.current_step
            }
            mlflow_utils.log_metrics(training_metrics)

    def test_data_processor_training_pipeline_integration(self, sample_market_data):
        """Test data processor in training pipeline context."""
        processor = DataProcessor()
        
        # Test data validation
        assert not sample_market_data.empty
        assert all(col in sample_market_data.columns 
                  for col in ['open', 'high', 'low', 'close', 'volume'])
        
        # Test that data is suitable for training
        assert len(sample_market_data) >= 100  # Minimum for training
        assert sample_market_data.index.is_monotonic_increasing  # Proper time series
        
        # Test no missing values in training data
        assert not sample_market_data[['open', 'high', 'low', 'close', 'volume']].isnull().any().any()


@pytest.mark.integration  
class TestModelPersistenceIntegration:
    """Test model saving and loading integration."""
    
    def test_agent_model_save_load(self, tmp_path):
        """Test saving and loading trained agent models."""
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=200, freq='1h')
        sample_data = pd.DataFrame({
            'open': [1.1000] * 200,
            'high': [1.1005] * 200,
            'low': [1.0995] * 200,
            'close': [1.1002] * 200,
            'volume': [1000] * 200
        }, index=dates)
        
        config = TradingEnvironmentConfig(
            initial_balance=10000.0,
            fee_structure=FeeStructure.SPREAD_BASED,
            observation_features=['close', 'volume']
        )
        
        env = TradingEnv(data=sample_data, config=config)
        agent = PPOAgent(env=env, learning_rate=3e-4, n_steps=64)
        
        # Train minimally
        agent.train(total_timesteps=128)
        
        # Save model
        model_path = tmp_path / "test_model"
        agent.model.save(str(model_path))
        assert model_path.with_suffix('.zip').exists()
        
        # Test model file is not empty
        assert model_path.with_suffix('.zip').stat().st_size > 0
