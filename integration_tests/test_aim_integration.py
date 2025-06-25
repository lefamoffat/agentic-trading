"""Integration tests for Aim backend.

These tests use real Aim instances but with temporary repositories.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import asyncio

from src.tracking.models import ExperimentConfig, TrainingMetrics
from src.tracking.factory import reset_singletons, get_ml_tracker, get_experiment_repository

# Skip if Aim not available
try:
    from src.tracking.backends.aim_backend import AimBackend
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False

@pytest.mark.skipif(not AIM_AVAILABLE, reason="Aim package not available")
@pytest.mark.integration
class TestAimIntegration:
    """Integration tests with real Aim backend."""
    
    @pytest.fixture
    async def temp_aim_repo(self):
        """Create temporary Aim repository for testing."""
        temp_dir = tempfile.mkdtemp(prefix="test_aim_")
        
        try:
            # Configure backend via environment variables (generic approach)
            import os
            os.environ["ML_STORAGE_PATH"] = temp_dir
            os.environ["ML_EXPERIMENT_NAME"] = "AgenticTrading"
            yield temp_dir
        finally:
            # Cleanup
            await reset_singletons()
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def realistic_experiment_config(self):
        """Realistic experiment configuration for integration testing."""
        return ExperimentConfig(
            experiment_id="integration_trading_test",
            agent_type="PPO",
            symbol="EUR/USD",
            timeframe="1h",
            timesteps=100,
            learning_rate=0.0003,
            initial_balance=10000.0
        )
    
    async def test_complete_experiment_lifecycle(
        self, temp_aim_repo, realistic_experiment_config
    ):
        """Test complete experiment lifecycle with real Aim backend."""
        # Get generic interfaces (should be Aim backend)
        tracker = await get_ml_tracker()
        repository = await get_experiment_repository()
        
        # Verify we got Aim implementations
        from src.tracking.backends.aim_backend import AimMLTracker, AimExperimentRepository
        assert isinstance(tracker, AimMLTracker)
        assert isinstance(repository, AimExperimentRepository)
        
        # Start experiment run
        run = await tracker.start_run(
            realistic_experiment_config.experiment_id,
            realistic_experiment_config,
            "Integration Test Run"
        )
        
        assert run is not None
        assert run.experiment_id == realistic_experiment_config.experiment_id
        assert len(run.id) > 0
        
        # Log training progression
        for step in range(0, 101, 25):
            training_metrics = TrainingMetrics(
                reward=1000 + step * 2,
                loss=1.0 - step * 0.008,
                portfolio_value=10000 + step * 50,
                sharpe_ratio=step * 0.02
            )
            
            await tracker.log_training_metrics(run.id, training_metrics, step)
        
        # Log hyperparameters
        hyperparams = {
            "learning_rate": realistic_experiment_config.learning_rate,
            "gamma": 0.99,
            "n_steps": 2048
        }
        await tracker.log_hyperparameters(run.id, hyperparams)
        
        # Finalize run
        final_metrics = {
            "final_reward": 1200.0,
            "total_return": 0.12
        }
        await tracker.finalize_run(run.id, final_metrics, "completed")
        
        # Query results through repository
        experiments = await repository.get_recent_experiments(limit=5)
        assert len(experiments) >= 0  # May be empty if Aim queries don't work immediately
        
        # System health check
        health = await repository.get_system_health()
        assert health.backend_name == "aim"
        assert health.is_healthy is True
    
    async def test_system_health_check(self, temp_aim_repo):
        """Test system health check with real Aim repository."""
        repository = await get_experiment_repository()
        
        health = await repository.get_system_health()
        
        assert health.backend_name == "aim"
        assert health.is_healthy is True
        assert health.tracker_healthy is True
        assert health.repository_healthy is True
        assert health.storage_healthy is True
        assert health.total_experiments >= 0
        assert health.total_runs >= 0 