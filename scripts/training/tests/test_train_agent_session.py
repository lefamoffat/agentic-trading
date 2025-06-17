import tempfile
from pathlib import Path
from types import SimpleNamespace

import mlflow
import pandas as pd
import pytest

from scripts.training import train_agent as ta

pytestmark = pytest.mark.component


class DummyModel:
    """A tiny stand-in for a stable-baselines3 model."""

    def __init__(self, *_, **__):
        self.saved_path = None

    def learn(self, *_, **__):
        return self

    def save(self, file_path):
        self.saved_path = Path(file_path)
        self.saved_path.parent.mkdir(parents=True, exist_ok=True)
        self.saved_path.write_text("dummy model file")


@pytest.fixture
def monkeypatched_train(monkeypatch):
    """Patch heavy dependencies inside train_agent.py to enable a fast smoke test."""

    monkeypatch.setattr(ta, "run_data_preparation_pipeline", lambda *_: True)

    df = pd.DataFrame({
        "open": [1, 2, 3, 4],
        "high": [1, 2, 3, 4],
        "low": [1, 2, 3, 4],
        "close": [1, 2, 3, 4],
        "volume": [10, 10, 10, 10],
    })
    monkeypatch.setattr(ta, "load_and_preprocess_data", lambda *_: (df, df))

    fake_env = SimpleNamespace()
    monkeypatch.setattr(ta.environment_factory, "create_environment", lambda *_, **__: fake_env)

    monkeypatch.setattr(ta, "MlflowMetricsCallback", lambda *_, **__: object())
    monkeypatch.setattr(ta, "GracefulShutdownCallback", lambda *_, **__: object())
    monkeypatch.setattr(ta, "CallbackList", lambda cb_list: cb_list)

    monkeypatch.setattr(ta.agent_factory, "create_agent", lambda *_, **__: DummyModel())

    yield


def test_train_agent_session_smoke(monkeypatched_train):
    with tempfile.TemporaryDirectory() as tmpdir:
        mlflow.set_tracking_uri(f"file://{tmpdir}")
        with mlflow.start_run() as run:
            ta.train_agent_session(
                agent_name="PPO",
                symbol="EUR/USD",
                timeframe="1h",
                timesteps=10,
                initial_balance=10000,
            )

        run_dir = Path(tmpdir) / run.info.experiment_id / run.info.run_id / "artifacts"
        assert run_dir.exists()

        model_root = Path("data/models/PPO") / run.info.run_id
        final_model = model_root / "final_model.zip"
        assert final_model.exists() 