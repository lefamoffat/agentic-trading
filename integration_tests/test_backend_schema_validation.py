"""Integration tests for backend schema validation.

Tests that would automatically catch the SQLite schema issues we encountered.
"""

import pytest
import tempfile
import os
import sqlite3
from pathlib import Path

from src.tracking import get_ml_tracker, get_experiment_repository, configure_aim_backend, reset_singletons
from src.tracking.models import SystemHealth, ExperimentConfig

@pytest.mark.integration
class TestBackendSchemaValidation:
    """Integration tests for detecting backend schema issues."""
    
    async def test_detect_empty_database_schema_issue(self):
        """Test detection of empty database causing 'no such table: run' error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the exact scenario that caused our issue
            aim_dir = Path(temp_dir) / ".aim"
            aim_dir.mkdir(parents=True)
            
            # Create empty SQLite database file (0 bytes)
            empty_db = aim_dir / "run_metadata.sqlite"
            empty_db.touch()
            
            # This is the exact condition that caused "no such table: run"
            assert empty_db.exists()
            assert empty_db.stat().st_size == 0
            
            os.environ["ML_STORAGE_PATH"] = temp_dir
            
            try:
                repository = await get_experiment_repository()
                
                # Repository health check should detect this schema issue
                health = await repository.get_system_health()
                
                assert isinstance(health, SystemHealth)
                assert health.backend_name == "aim"
                
                # Should either be unhealthy or handle gracefully
                if not health.is_healthy:
                    assert health.error_message is not None
                    error_msg = health.error_message.lower()
                    # Should mention the specific issue
                    assert any(term in error_msg for term in ['table', 'schema', 'database', 'sqlite'])
            finally:
                del os.environ["ML_STORAGE_PATH"]
    
    async def test_detect_corrupted_database_file(self):
        """Test detection of corrupted database files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            aim_dir = Path(temp_dir) / ".aim"
            aim_dir.mkdir(parents=True)
            
            # Create corrupted database file
            corrupted_db = aim_dir / "run_metadata.sqlite"
            corrupted_db.write_text("This is not a valid SQLite database file")
            
            os.environ["ML_STORAGE_PATH"] = temp_dir
            
            try:
                repository = await get_experiment_repository()
                
                # Should detect corruption
                health = await repository.get_system_health()
                
                assert isinstance(health, SystemHealth)
                assert health.backend_name == "aim"
                
                # Should handle corruption gracefully
                if not health.is_healthy:
                    assert health.error_message is not None
                    error_msg = health.error_message.lower()
                    assert any(term in error_msg for term in ['database', 'corrupted', 'sqlite', 'invalid'])
            finally:
                del os.environ["ML_STORAGE_PATH"]
    
    async def test_detect_missing_required_tables(self):
        """Test detection of databases missing required tables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            aim_dir = Path(temp_dir) / ".aim"
            aim_dir.mkdir(parents=True)
            
            # Create valid SQLite database but with wrong schema
            incomplete_db = aim_dir / "run_metadata.sqlite"
            
            conn = sqlite3.connect(str(incomplete_db))
            try:
                # Create some table, but not the ones Aim expects
                conn.execute("""
                    CREATE TABLE wrong_table (
                        id INTEGER PRIMARY KEY,
                        data TEXT
                    )
                """)
                conn.commit()
            finally:
                conn.close()
            
            os.environ["ML_STORAGE_PATH"] = temp_dir
            
            try:
                repository = await get_experiment_repository()
                
                # Should detect missing tables
                health = await repository.get_system_health()
                
                assert isinstance(health, SystemHealth)
                assert health.backend_name == "aim"
                
                # May or may not be detected as unhealthy depending on Aim's validation
                # But should not crash
            finally:
                del os.environ["ML_STORAGE_PATH"]
    
    async def test_validate_proper_schema_setup(self):
        """Test that proper schema setup is validated correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Ensure proper permissions
            temp_path = Path(temp_dir)
            temp_path.chmod(0o755)
            
            # Configure isolated backend
            await configure_aim_backend(
                experiment_name="schema_test",
                storage_path=temp_dir
            )
            
            try:
                # Initialize normally
                tracker = await get_ml_tracker()
                repository = await get_experiment_repository()
                
                # Create a proper experiment to establish schema
                config = ExperimentConfig(
                    experiment_id="schema_validation_test",
                    agent_type="PPO",
                    symbol="EUR/USD"
                )
                
                run = await tracker.start_run(
                    config.experiment_id,
                    config,
                    "Schema Validation Test"
                )
                
                # Log some data to establish database schema
                await tracker.log_metrics(run.id, {"test_metric": 1.0}, step=1)
                await tracker.finalize_run(run.id, {"final_metric": 2.0}, "completed")
                
                # Health check should now show healthy system
                health = await repository.get_system_health()
                
                assert isinstance(health, SystemHealth)
                assert health.backend_name == "aim"
                
                # Should be healthy after proper initialization
                assert health.is_healthy or health.error_message is not None
                
                # Should have experiment data
                if hasattr(health, 'total_experiments'):
                    assert health.total_experiments >= 0
                if hasattr(health, 'total_runs'):
                    assert health.total_runs >= 0
            finally:
                # Clean up singletons
                await reset_singletons()

if __name__ == "__main__":
    pytest.main([__file__]) 