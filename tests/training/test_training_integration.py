import os
import shutil
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import mlflow
import pytest
import pandas as pd

# Import the main and objective functions from the scripts
from scripts.training.train_agent import main as train_agent_main
from scripts.training.optimize_agent import objective, main as optimize_agent_main


@pytest.fixture(scope="module")
def project_root():
    """Return the project root directory."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture
def training_env_setup(tmp_path, project_root, monkeypatch):
    """Set up the environment for a training integration test."""
    monkeypatch.chdir(tmp_path)

    # Set up MLflow to use a temporary directory
    mlflow_dir = tmp_path / "mlruns"
    mlflow_dir.mkdir()
    mlflow.set_tracking_uri(mlflow_dir.as_uri())
    experiment_name = "test_training_integration"
    mlflow.set_experiment(experiment_name)
    
    # Copy necessary configs
    shutil.copytree(project_root / "config", tmp_path / "config")
    
    # Create the final processed feature file that the script expects to find
    # after the (mocked) data preparation pipeline has run.
    processed_dir = tmp_path / "data" / "processed" / "features"
    processed_dir.mkdir(parents=True, exist_ok=True)
    dummy_df = pd.DataFrame({
        'timestamp': pd.to_datetime(pd.date_range(start='1/1/2022', periods=200)),
        'open': 1.0, 'high': 1.1, 'low': 0.9, 'close': 1.05, 'volume': 100,
        'feature_1': 0.5, 'feature_2': -0.5
    })
    # Use a valid timeframe like '1d'
    dummy_df.to_csv(processed_dir / "test_symbol_1d_features.csv", index=False)

    yield experiment_name

    # Teardown
    mlflow.set_tracking_uri(None)


@pytest.mark.integration
@patch("scripts.training.train_agent.run_data_preparation_pipeline", return_value=True)
def test_train_agent_cli_integration(mock_pipeline, training_env_setup, monkeypatch):
    """
    Test the training script's CLI as a full integration run.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/training/train_agent.py",
            "--symbol", "test_symbol",
            "--timeframe", "1d", # Use valid timeframe
            "--timesteps", "100",
            "--agent", "PPO",
            "--balance", "10000"
        ],
    )

    # Call the actual main function from the script
    train_agent_main()
            
    experiment = mlflow.get_experiment_by_name(training_env_setup)
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
    run = runs.iloc[0]
    assert run.status == "FINISHED"
    assert int(run["params.timesteps"]) == 100
    assert run["params.agent"] == "PPO"


@pytest.mark.integration
@patch("scripts.training.train_agent.run_data_preparation_pipeline", return_value=True)
def test_objective_function_integration(mock_pipeline, training_env_setup):
    """
    Test the Optuna objective function as an integration test.
    """
    with mlflow.start_run(run_name="test_parent_run") as parent_run:
        parent_run_id = parent_run.info.run_id
        
        trial = MagicMock()
        trial.suggest_categorical.return_value = 32
        trial.suggest_int.return_value = 16
        trial.suggest_float.return_value = 0.0003
        trial.number = 0

        with patch("mlflow.tracking.MlflowClient.get_metric_history", return_value=[MagicMock(value=1.5)]):
            result = objective(
                trial=trial,
                agent_name="PPO",
                symbol="test_symbol",
                timeframe="1d", # Use valid timeframe
                timesteps=100,
                initial_balance=10000,
                parent_run_id=parent_run_id,
            )

    assert isinstance(result, float)
    assert result == 1.5

    experiment = mlflow.get_experiment_by_name(training_env_setup)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time"])
    
    assert len(runs) == 2 
    nested_run = runs[runs["tags.mlflow.parentRunId"] == parent_run_id].iloc[0]
    assert nested_run is not None
    assert nested_run.status == "FINISHED"
    assert nested_run["tags.mlflow.runName"] == "trial_0"


@pytest.mark.integration
@patch("scripts.training.train_agent.run_data_preparation_pipeline", return_value=True)
def test_optimize_agent_cli_integration(mock_pipeline, training_env_setup, monkeypatch):
    """
    Test the optimize_agent CLI script as a full integration run.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/training/optimize_agent.py",
            "--symbol", "test_symbol",
            "--timeframe", "1d", # Use valid timeframe
            "--timesteps", "100",
            "--trials", "2",
            "--agent", "PPO"
        ],
    )
    
    with patch("mlflow.tracking.MlflowClient.get_metric_history", return_value=[MagicMock(value=1.5)]):
        # Call the actual main function from the script
        optimize_agent_main()

    experiment = mlflow.get_experiment_by_name(training_env_setup)
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    assert len(runs) == 3 
    parent_runs = runs[runs["tags.mlflow.runName"].str.startswith("HPO_PPO_")]
    assert len(parent_runs) == 1
    
    parent_run_id = parent_runs.iloc[0].run_id
    child_runs = runs[runs["tags.mlflow.parentRunId"] == parent_run_id]
    assert len(child_runs) == 2 