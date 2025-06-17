import pandas as pd
import tempfile
from pathlib import Path
from types import SimpleNamespace
import pytest

from scripts.analysis import backend as backend_mod

# Dummy SimulationResult
class _DummyResult(SimpleNamespace):
    pass

@pytest.mark.unit
def test_run_simulation_drops_nans(monkeypatch):
    # Create a temp CSV with NaNs
    tmp_dir = tempfile.TemporaryDirectory()
    csv_path = Path(tmp_dir.name) / "feat.csv"
    df = pd.DataFrame({"close": [1.0, float('nan'), 2.0], "feat1": [0.1, 0.2, float('nan')]})
    df.to_csv(csv_path, index=False)

    # Patch Backtester so we can inspect the dataframe it receives
    captured = {}

    class _DummyBacktester:
        def __init__(self, *, model_uri: str, data: pd.DataFrame, initial_balance: float):
            # Assert no NaNs in incoming data
            assert not data.isna().any().any()
            captured["rows"] = len(data)
            captured["uri"] = model_uri

        def run(self):
            return _DummyResult()

    monkeypatch.setattr(backend_mod, "Backtester", _DummyBacktester)

    result = backend_mod.run_simulation(
        model_uri="models:/dummy@1",
        data_path=str(csv_path),
        initial_balance=10000.0,
    )

    # Ensure backtester run occurred and NaNs were removed (should be 1 row left)
    assert isinstance(result, _DummyResult)
    assert captured["rows"] == 1
    tmp_dir.cleanup() 