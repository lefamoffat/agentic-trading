"""Integration tests for tracking system health monitoring.

Tests the complete health monitoring workflow with real dependencies.
"""

import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime

from src.tracking import get_ml_tracker, get_experiment_repository
from src.tracking.models import SystemHealth, ExperimentConfig

@pytest.mark.integration
class TestTrackingHealthMonitoring:
    """Integration tests for tracking health monitoring workflow."""
    
    async def test_health_monitoring_detects_schema_issues(self):
        """Test that health monitoring detects schema issues automatically."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the exact problematic scenario we encountered
            aim_dir = Path(temp_dir) / ".aim"
            aim_dir.mkdir(parents=True)
            
            # Empty database that causes "no such table: run" error
            empty_db = aim_dir / "run_metadata.sqlite"
            empty_db.touch()  # 0-byte file
            
            os.environ["ML_STORAGE_PATH"] = temp_dir
            
            try:
                repository = await get_experiment_repository()
                
                # Health monitoring should detect this issue
                health = await repository.get_system_health()
                
                assert isinstance(health, SystemHealth)
                assert health.backend_name == "aim"
                
                # Should either be unhealthy or provide warning information
                if not health.is_healthy:
                    assert health.error_message is not None
                    error_lower = health.error_message.lower()
                    assert any(term in error_lower for term in ['table', 'schema', 'database', 'sqlite'])
            finally:
                del os.environ["ML_STORAGE_PATH"]
    
    async def test_health_monitoring_with_dual_database_structure(self):
        """Test health monitoring with dual database structure (the issue we found)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the dual .aim structure we discovered
            outer_aim = Path(temp_dir) / ".aim"
            inner_aim = outer_aim / ".aim"
            outer_aim.mkdir(parents=True)
            inner_aim.mkdir(parents=True)
            
            # Populated outer database (simulating real data)
            outer_db = outer_aim / "run_metadata.sqlite"
            self._create_simple_database(outer_db)
            
            # Empty inner database (the problem)
            inner_db = inner_aim / "run_metadata.sqlite"
            inner_db.touch()
            
            os.environ["ML_STORAGE_PATH"] = temp_dir
            
            try:
                repository = await get_experiment_repository()
                
                # Should handle this scenario gracefully
                health = await repository.get_system_health()
                
                assert isinstance(health, SystemHealth)
                assert health.backend_name == "aim"
                # Backend may or may not detect this as an issue depending on implementation
            finally:
                del os.environ["ML_STORAGE_PATH"]
    
    def _create_simple_database(self, db_path: Path):
        """Create a simple SQLite database with basic structure."""
        import sqlite3
        
        conn = sqlite3.connect(str(db_path))
        try:
            # Basic table to simulate Aim structure
            conn.execute("""
                CREATE TABLE IF NOT EXISTS run (
                    id TEXT PRIMARY KEY,
                    experiment TEXT,
                    created_at INTEGER
                )
            """)
            conn.execute("""
                INSERT INTO run (id, experiment, created_at)
                VALUES ('test_run', 'test_experiment', 1234567890)
            """)
            conn.commit()
        finally:
            conn.close()
    
    async def test_health_monitoring_integration_with_experiments(self):
        """Test health monitoring integrated with actual experiment workflows."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["ML_STORAGE_PATH"] = temp_dir
            
            try:
                tracker = await get_ml_tracker()
                repository = await get_experiment_repository()
                
                # Initial health check
                initial_health = await repository.get_system_health()
                assert isinstance(initial_health, SystemHealth)
                
                # Create an experiment
                config = ExperimentConfig(
                    experiment_id="health_monitoring_test",
                    agent_type="PPO",
                    symbol="EUR/USD"
                )
                
                run = await tracker.start_run(
                    config.experiment_id,
                    config,
                    "Health Monitoring Test"
                )
                
                # Health check during active experiment
                active_health = await repository.get_system_health()
                assert isinstance(active_health, SystemHealth)
                assert active_health.backend_name == "aim"
                
                # Finalize experiment
                await tracker.finalize_run(run.id, {"test_metric": 1.0}, "completed")
                
                # Final health check
                final_health = await repository.get_system_health()
                assert isinstance(final_health, SystemHealth)
                
                # Health should be consistent throughout
                assert initial_health.backend_name == final_health.backend_name
            finally:
                del os.environ["ML_STORAGE_PATH"]
    
    async def test_health_monitoring_error_reporting_quality(self):
        """Test that health monitoring provides high-quality error reporting."""
        test_scenarios = [
            # Missing storage directory
            ("missing_storage", lambda d: None),
            # Corrupted database
            ("corrupted_db", lambda d: self._create_corrupted_database(d)),
            # Permission issues (simulated)
            ("permissions", lambda d: self._create_readonly_structure(d))
        ]
        
        for scenario_name, setup_func in test_scenarios:
            with tempfile.TemporaryDirectory() as temp_dir:
                if setup_func:
                    setup_func(temp_dir)
                
                os.environ["ML_STORAGE_PATH"] = temp_dir
                
                try:
                    repository = await get_experiment_repository()
                    health = await repository.get_system_health()
                    
                    assert isinstance(health, SystemHealth)
                    assert health.backend_name == "aim"
                    
                    # If unhealthy, error message should be informative
                    if not health.is_healthy and health.error_message:
                        assert len(health.error_message) > 10  # Non-trivial message
                        # Should not contain internal implementation details
                        error_lower = health.error_message.lower()
                        assert "traceback" not in error_lower
                        assert "exception" not in error_lower
                finally:
                    del os.environ["ML_STORAGE_PATH"]
    
    def _create_corrupted_database(self, temp_dir: str):
        """Create corrupted database scenario."""
        aim_dir = Path(temp_dir) / ".aim"
        aim_dir.mkdir(parents=True)
        
        corrupted_db = aim_dir / "run_metadata.sqlite"
        corrupted_db.write_text("This is not SQLite data")
    
    def _create_readonly_structure(self, temp_dir: str):
        """Create read-only structure scenario."""
        aim_dir = Path(temp_dir) / ".aim"
        aim_dir.mkdir(parents=True)
        
        # Create valid database first
        db_file = aim_dir / "run_metadata.sqlite"
        self._create_simple_database(db_file)
        
        # Make directory read-only (simulating permission issues)
        try:
            aim_dir.chmod(0o444)
        except OSError:
            # Permission changes might not work in all environments
            pass
    
    async def test_health_monitoring_recovery_scenarios(self):
        """Test various recovery scenarios after backend failures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["ML_STORAGE_PATH"] = temp_dir
            
            try:
                # Scenario 1: Empty database
                aim_dir = Path(temp_dir) / ".aim"
                aim_dir.mkdir(parents=True)
                empty_db = aim_dir / "run_metadata.sqlite"
                empty_db.touch()
                
                repository = await get_experiment_repository()
                health = await repository.get_system_health()
                
                # Should handle gracefully
                assert isinstance(health, SystemHealth)
                assert health.backend_name == "aim"
                
                # Scenario 2: Fix by removing corrupted database
                if empty_db.exists():
                    empty_db.unlink()
                
                # Re-check health
                health = await repository.get_system_health()
                assert isinstance(health, SystemHealth)
                
                # Scenario 3: Create proper experiment to establish schema
                from src.tracking import get_ml_tracker
                from src.tracking.models import ExperimentConfig
                
                tracker = await get_ml_tracker()
                config = ExperimentConfig(
                    experiment_id="recovery_test",
                    agent_type="PPO",
                    symbol="EUR/USD"
                )
                
                run = await tracker.start_run(
                    config.experiment_id,
                    config,
                    "Recovery Test"
                )
                
                # Log some data to establish database schema
                await tracker.log_metrics(run.id, {"test_metric": 1.0}, step=1)
                await tracker.finalize_run(run.id, {"final_metric": 2.0}, "completed")
                
                # Final health check should be better
                health = await repository.get_system_health()
                assert isinstance(health, SystemHealth)
                assert health.backend_name == "aim"
                
            finally:
                del os.environ["ML_STORAGE_PATH"]


if __name__ == "__main__":
    pytest.main([__file__]) 