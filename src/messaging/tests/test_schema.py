"""Unit tests for src.messaging.schema.ExperimentState utilities."""

import json
import pytest
from time import time

from src.messaging.schema import ExperimentState


class TestExperimentStateModel:
    """Validate ExperimentState model behaviour."""

    def test_status_field_lowercasing(self):
        """Status should be converted to lower-case regardless of input case."""
        state = ExperimentState(
            id="exp_001",
            status="RUNNING",  # mixed case
            config={},
            start_time=time(),
        )
        assert state.status == "running"

    def test_to_flat_dict_serialisation(self):
        """to_flat_dict() should convert nested / numeric values to JSON-compatible str mapping."""
        metrics = {"reward": 1.23, "loss": 0.45}
        cfg = {"algo": "ppo", "lr": 0.001}
        now = time()
        state = ExperimentState(
            id="exp_002",
            status="completed",
            config=cfg,
            start_time=now,
            end_time=now + 10,
            current_step=100,
            total_steps=100,
            metrics=metrics,
            duration=10.0,
            message=None,
            last_updated=now + 10,
        )

        flat = state.to_flat_dict()

        # All values must be str
        assert all(isinstance(v, str) for v in flat.values())
        # Dicts must be JSON-encoded strings
        assert json.loads(flat["config"]) == cfg
        assert json.loads(flat["metrics"]) == metrics
        # None converted to literal "None"
        assert flat["message"] == "None"
        # Numeric values converted via str()
        assert flat["current_step"] == "100"
        assert flat["duration"] == "10.0"

    def test_from_redis_hash_roundtrip(self):
        """from_redis_hash() must invert to_flat_dict() accurately."""
        state_original = ExperimentState(
            id="exp_003",
            status="failed",
            config={"algo": "dqn"},
            start_time=123.456,
            end_time=130.0,
            duration=6.544,
            current_step=50,
            total_steps=500,
            metrics={"reward": -1.0},
            message="error",
            last_updated=130.0,
        )

        flat = state_original.to_flat_dict()
        reconstructed = ExperimentState.from_redis_hash(flat)

        assert reconstructed == state_original 