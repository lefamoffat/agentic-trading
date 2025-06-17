import builtins 
import types
from types import SimpleNamespace
from typing import Any
import pytest
import mlflow

from src.utils.mlflow import ensure_experiment, log_sb3_model


class _DummyClient:
    """Minimal stub to stand in for ``mlflow.tracking.MlflowClient``."""

    def __init__(self):
        self._created_name: str | None = None
        self._get_called: bool = False
        self._exp = SimpleNamespace(experiment_id="42")

    # Scenario toggle injected by tests
    should_exist: bool = False

    # ---- API stubs -----------------------------------------------------
    def get_experiment_by_name(self, name: str):
        self._get_called = True
        return self._exp if self.should_exist else None

    def create_experiment(self, *, name: str):  # noqa: D401 (simple signature)
        self._created_name = name
        return "99"


@pytest.mark.unit
def test_ensure_experiment_creates_when_missing(monkeypatch):
    client = _DummyClient()
    client.should_exist = False

    monkeypatch.setattr("src.utils.mlflow.mlflow.tracking.MlflowClient", lambda *a, **k: client)
    called: dict[str, Any] = {}
    monkeypatch.setattr("src.utils.mlflow.mlflow.set_experiment", lambda name: called.setdefault("name", name))

    exp_id = ensure_experiment("MyExperiment")

    assert exp_id == "99"
    assert client._created_name == "MyExperiment"
    assert called["name"] == "MyExperiment"


@pytest.mark.unit
def test_ensure_experiment_uses_existing(monkeypatch):
    client = _DummyClient()
    client.should_exist = True

    monkeypatch.setattr("src.utils.mlflow.mlflow.tracking.MlflowClient", lambda *a, **k: client)
    called: dict[str, Any] = {}
    monkeypatch.setattr("src.utils.mlflow.mlflow.set_experiment", lambda name: called.setdefault("name", name))

    exp_id = ensure_experiment("Existing")

    assert exp_id == "42"  # returned from dummy exp
    assert client._created_name is None  # no create_experiment call
    assert called["name"] == "Existing"


@pytest.mark.unit
def test_log_sb3_model_registers(monkeypatch):
    # Stub the mlflow.pyfunc.log_model response
    dummy_logged = SimpleNamespace(model_uri="runs:/dummy/abcd")
    monkeypatch.setattr("src.utils.mlflow.mlflow.pyfunc.log_model", lambda **kwargs: dummy_logged)

    captured: dict[str, Any] = {}

    def _register_model(model_uri: str, name: str):
        captured["uri"] = model_uri
        captured["name"] = name
        return SimpleNamespace()

    monkeypatch.setattr("src.utils.mlflow.mlflow.register_model", _register_model)

    # Call under test
    uri = log_sb3_model(
        model=object(),
        name="TestModel",
        artifacts={"model_path": "/tmp/model.zip"},
        signature_df=SimpleNamespace(),
        python_model=SimpleNamespace(),
    )

    assert uri == "runs:/dummy/abcd"
    assert captured["uri"] == "runs:/dummy/abcd"
    assert captured["name"] == "TestModel"


# ---------------------------------------------------------------------------
# log_metrics sanitisation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_log_metrics_sanitises(monkeypatch):
    captured = {}

    def _log_metrics(metrics, **kwargs):
        captured.update(metrics)

    monkeypatch.setattr("src.utils.mlflow.mlflow.log_metrics", _log_metrics)

    from src.utils.mlflow import log_metrics as log_metrics_fn

    bad_metrics = {
        "good": 1.23,
        "nan_val": float('nan'),
        "inf_val": float('inf'),
        "str": "oops",
    }

    log_metrics_fn(bad_metrics)

    assert captured == {"good": 1.23} 