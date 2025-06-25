#!/usr/bin/env python3
"""Integration tests for infrastructure components.

These tests verify that the various infrastructure components work together
correctly, including ML tracking, data processing, and system initialization.
"""
import os
import sys
import tempfile
from pathlib import Path
import pytest
import pandas as pd

from unittest.mock import patch


from src.utils.config_loader import ConfigLoader
from src.market_data import DataProcessor
from src.environment.config import TradingEnvironmentConfig, FeeStructure
from src.environment import load_trading_config

# Add src to path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

@pytest.mark.integration
class TestInfrastructureIntegration:
    """Integration tests for infrastructure components without external dependencies."""
    
    def test_project_structure_integrity(self):
        """Test that essential project structure exists."""
        # Check critical directories exist
        assert Path("src").exists()
        assert Path("configs").exists()
        assert Path("apps").exists()
        
        # Check critical config files exist
        assert Path("configs/trading_config.yaml").exists()
        assert Path("configs/agent_config.yaml").exists()
        
        # Check entry points exist
        assert Path("apps/cli").exists()
        assert Path("apps/dashboard").exists()
        
    def test_configuration_loading(self):
        """Test configuration files can be loaded without errors."""
        from src.environment import load_trading_config
        from pathlib import Path
        
        # Test loading trading config
        config_path = Path("configs/trading_config.yaml")
        trading_config = load_trading_config(config_path)
        assert trading_config is not None
        assert hasattr(trading_config, 'initial_balance')
        
    def test_logging_system_initialization(self):
        """Test logging system can be initialized."""
        from src.utils.logger import get_logger
        
        logger = get_logger("test_logger")
        assert logger is not None
        
        # Test basic logging operations
        logger.info("Test log message")
        logger.debug("Test debug message")
        logger.warning("Test warning message")

@pytest.mark.integration 
class TestMLTrackingIntegration:
    """Integration tests for ML tracking system."""
    
    @pytest.mark.asyncio
    async def test_tracking_system_initialization(self):
        """Test that ML tracking system can be initialized."""
        try:
            from src.tracking import get_ml_tracker, get_experiment_repository
            
            # Test tracker initialization
            tracker = await get_ml_tracker()
            assert tracker is not None
            
            # Test repository initialization  
            repository = await get_experiment_repository()
            assert repository is not None
            
            # Test health check
            health = await repository.get_system_health()
            assert health is not None
            assert health.backend_name == "aim"
            
        except ImportError:
            pytest.skip("ML tracking not available")
    
    @pytest.mark.asyncio
    async def test_basic_experiment_lifecycle(self):
        """Test basic experiment creation and tracking."""
        try:
            from src.tracking import get_ml_tracker
            from src.tracking.models import ExperimentConfig, TrainingMetrics
            
            tracker = await get_ml_tracker()
            
            # Create experiment config
            config = ExperimentConfig(
                experiment_id="test_integration_experiment",
                agent_type="PPO",
                symbol="EUR/USD",
                timesteps=1000
            )
            
            # Start run
            run = await tracker.start_run("test_integration_experiment", config)
            assert run is not None
            assert run.experiment_id == "test_integration_experiment"
            
            # Log some metrics
            metrics = TrainingMetrics(
                reward=100.5,
                portfolio_value=10500.0,
                win_rate=0.65
            )
            await tracker.log_training_metrics(run.id, metrics, step=1)
            
            # Finalize run
            await tracker.finalize_run(run.id, {"final_reward": 150.0}, "completed")
            
        except ImportError:
            pytest.skip("ML tracking not available")

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
        processor = DataProcessor(symbol="EUR/USD", asset_class="forex")
        
        # Test that the processor can handle basic data operations
        assert processor is not None
        assert processor.symbol == "EUR/USD"
        assert processor.asset_class == "forex"
        
        # Test basic data validation
        assert not sample_market_data.empty
        assert all(col in sample_market_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    def test_environment_data_integration(self, sample_market_data):
        """Test that processed data can be used in trading environment."""
        from src.environment import TradingEnv
        
        config = TradingEnvironmentConfig(
            initial_balance=10000.0,
            fee_structure=FeeStructure.SPREAD_BASED,
            observation_features=['close', 'volume'],
            include_time_features=False  # Disable time features for simpler test
        )
        
        # Test environment can be created with processed data
        env = TradingEnv(data=sample_market_data, config=config)
        assert env is not None
        
        # Test basic environment operations
        obs, info = env.reset()
        assert obs is not None
        assert len(obs) == 6  # 2 market features + 4 portfolio features (no time features)
        
        # Test environment step
        obs, reward, terminated, truncated, info = env.step(0)  # No-op action
        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
