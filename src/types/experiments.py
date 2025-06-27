"""Canonical experiment data models used across the whole code-base.

Every layer (training, messaging, API, CLI, dashboard) must exchange these
models ‑ never plain dictionaries.  There is *zero* legacy compatibility: any
code still accessing ``exp["id"]`` etc. will fail mypy or at runtime.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field, model_validator


class TrainingStatus(str, Enum):
    """Lifecycle states for an experiment (mirrors messaging events)."""

    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentConfig(BaseModel):
    """Launch-time configuration parameters sent by the user."""

    agent_type: str
    symbol: str
    timeframe: str
    timesteps: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    initial_balance: float = Field(gt=0)


class ExperimentState(BaseModel):
    """Dynamic runtime fields updated during training."""

    status: TrainingStatus = TrainingStatus.STARTING
    start_time: float  # epoch seconds
    end_time: Optional[float] = None
    duration: Optional[float] = None
    current_step: int = 0
    total_steps: int = 0
    metrics: Dict[str, Any] = Field(default_factory=dict)
    message: Optional[str] = None

    @model_validator(mode="after")
    def _validate_steps(self) -> "ExperimentState":  # noqa: D401 (simple)
        if self.current_step > self.total_steps and self.total_steps > 0:
            raise ValueError("current_step cannot exceed total_steps")
        return self


class Experiment(BaseModel):
    """Full experiment entity (immutable view)."""

    id: str  # canonical identifier
    config: ExperimentConfig
    state: ExperimentState

    # ------------------------------------------------------------------
    # Convenience computed properties
    # ------------------------------------------------------------------
    @property
    def status(self) -> TrainingStatus:  # noqa: D401 (simple)
        return self.state.status

    @property
    def progress(self) -> float:  # 0-1
        if self.state.total_steps == 0:
            return 0.0
        return self.state.current_step / self.state.total_steps

    # ------------------------------------------------------------------
    # Serialisation helpers (used by FastAPI / dashboard / CLI)
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:  # noqa: D401 (simple)
        """Return a JSON-serialisable representation (flat dict)."""
        doc = self.model_dump(mode="python", by_alias=True)
        # Flatten nested models for easy front-end access
        flattened_config = doc.pop("config")
        flattened_state = doc.pop("state")

        doc.update(flattened_config)
        doc.update(flattened_state)

        # ------------------------------------------------------------------
        # Canonical aliases expected by API & CLI payload contracts
        # ------------------------------------------------------------------
        doc["experiment_id"] = self.id  # human-friendly alias
        doc["config"] = flattened_config  # keep full nested config for strict clients

        # Normalise enum values for clean JSON --------------------------------
        for key, value in doc.items():
            if isinstance(value, Enum):
                doc[key] = value.value

        return doc

    # ------------------------------------------------------------------
    # Redis serialisation helpers – single source of truth for the
    # messaging layer.  No other module should perform ad-hoc flattening
    # or type casting – always go through these helpers to guarantee
    # consistency across CLI, dashboard and worker processes.
    # ------------------------------------------------------------------
    def to_redis_hash(self) -> Dict[str, str]:
        """Return a flat mapping ready for ``redis.hset``.

        * Values are converted to ``str``.
        * ``None`` values are **omitted** – they are not stored as the
          literal string ``"None"`` to avoid deserialisation errors.
        * Nested dictionaries (e.g. ``metrics``) are JSON-encoded.
        """
        import json  # local import to avoid heavy dependency when not used

        # Flatten config & state – ExperimentConfig and ExperimentState are
        # Pydantic models, so ``model_dump`` gives plain Python types.
        flat: Dict[str, Any] = {
            **self.config.model_dump(mode="python"),
            **self.state.model_dump(mode="python"),
            "id": self.id,
            "status": self.state.status.value,
        }

        mapping: Dict[str, str] = {}
        for k, v in flat.items():
            if v is None:
                # We deliberately *skip* ``None`` values instead of
                # storing a placeholder string.  This keeps the Redis
                # hash clean and makes type conversion unambiguous.
                continue
            if isinstance(v, dict):
                mapping[k] = json.dumps(v, separators=(",", ":"))
            else:
                mapping[k] = str(v)
        return mapping

    @staticmethod
    def from_redis_hash(data: Dict[str, str]) -> "Experiment":
        """Reconstruct an ``Experiment`` from a Redis ``HGETALL`` result.

        The helper is **tolerant** to historical hashes that might still
        contain the string ``"None"``.
        """
        import json  # local import to avoid heavy dependency when not used

        def _as_int(value: str | None) -> int:
            if value in (None, "", "None"):
                return 0
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        def _as_float(value: str | None) -> float | None:
            if value in (None, "", "None"):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _as_json(value: str | None) -> Dict[str, Any]:
            if value in (None, "", "None"):
                return {}
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {}

        cfg = ExperimentConfig(
            agent_type=data.get("agent_type", ""),
            symbol=data.get("symbol", ""),
            timeframe=data.get("timeframe", ""),
            timesteps=_as_int(data.get("timesteps")),
            learning_rate=float(data.get("learning_rate", 0)) if data.get("learning_rate") not in ("", "None", None) else 0.0,
            initial_balance=float(data.get("initial_balance", 0)) if data.get("initial_balance") not in ("", "None", None) else 0.0,
        )

        state = ExperimentState(
            status=TrainingStatus(data.get("status", TrainingStatus.STARTING.value)),
            start_time=float(data.get("start_time", 0.0)) if data.get("start_time") not in ("", "None", None) else 0.0,
            end_time=_as_float(data.get("end_time")),
            duration=_as_float(data.get("duration")),
            current_step=_as_int(data.get("current_step")),
            total_steps=_as_int(data.get("total_steps")),
            metrics=_as_json(data.get("metrics")),
            message=data.get("message") if data.get("message") not in ("None", "") else None,
        )

        return Experiment(id=data.get("id", ""), config=cfg, state=state) 