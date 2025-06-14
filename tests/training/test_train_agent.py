#!/usr/bin/env python3
"""
Integration tests for the agent training script.
"""
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from scripts.training.train_agent import train_agent
from src.agents.factory import agent_factory
from src.environments.factory import environment_factory
from src.utils.config_loader import ConfigLoader


@pytest.fixture
def training_env_setup(monkeypatch):
    """
    Set up the environment for a training integration test.
    - Creates a dummy feature file.
    - Mocks the feature generation subprocess.
    - Cleans up created directories and files afterward.
    """
    # 1. Setup paths and unique run_id
    run_id = f"test_run_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    symbol = "TESTSYMBOL"
    agent_name = "PPO"
    
    base_data_dir = Path("data")
    features_dir = base_data_dir / "processed" / "features"
    models_dir = base_data_dir / "models" / agent_name / run_id
    logs_dir = Path("logs") / "tensorboard" / agent_name / run_id
    
    # Create dummy feature file
    features_dir.mkdir(parents=True, exist_ok=True)
    dummy_features_path = features_dir / f"{symbol}_1h_features.csv"
    
    data = {
        "timestamp": pd.to_datetime([f"2023-01-01 {i:02d}:00" for i in range(20)]),
        "open": [100 + i for i in range(20)],
        "high": [102 + i for i in range(20)],
        "low": [99 + i for i in range(20)],
        "close": [101 + i for i in range(20)],
        "volume": [1000] * 20,
    }
    pd.DataFrame(data).to_csv(dummy_features_path, index=False)
    
    # 2. Mock the feature generation subprocess
    def mock_subprocess_run(*args, **kwargs):
        print("Mocked subprocess.run called, skipping feature generation.")
        return subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)
    
    yield symbol, run_id, models_dir, logs_dir
    
    # 3. Teardown
    print("Tearing down test environment...")
    shutil.rmtree(models_dir, ignore_errors=True)
    shutil.rmtree(logs_dir, ignore_errors=True)
    shutil.rmtree(features_dir, ignore_errors=True)


@pytest.mark.unit
def test_train_agent_integration(training_env_setup, monkeypatch):
    """
    Test the full training pipeline for a small number of timesteps.
    """
    symbol, run_id, models_dir, logs_dir = training_env_setup

    # 1. Prepare a modified config for the test
    config_loader = ConfigLoader()
    agent_config = config_loader.load_config("agent_config")
    agent_config["ppo"] = agent_config.get("ppo", {})
    agent_config["ppo"]["n_steps"] = 8
    agent_config["ppo"]["batch_size"] = 8
    agent_config["training"]["eval_frequency"] = 5

    # 2. Monkeypatch the config loader to return the modified config
    def mock_load_config(self, config_name):
        print(f"Mocked load_config for '{config_name}'. Returning modified config.")
        return agent_config
    
    monkeypatch.setattr(ConfigLoader, "load_config", mock_load_config)

    # 3. Run the training process
    train_agent(
        agent_name="PPO",
        symbol=symbol,
        timeframe="1h",
        total_timesteps=16,  # Must be a multiple of n_steps
        run_id=run_id,
    )

    # 4. Verify that model and log files were created
    assert models_dir.exists()
    assert (models_dir / "best_model.zip").exists()
    assert (models_dir / "final_model.zip").exists()
    
    assert logs_dir.exists()
    # Check for at least one TensorBoard event file recursively
    event_files = list(logs_dir.rglob("events.out.tfevents.*"))
    assert len(event_files) > 0, "No TensorBoard event files were found." 