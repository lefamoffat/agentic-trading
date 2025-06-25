from __future__ import annotations

"""Typed schemas used by the messaging layer."""
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
import json


class ExperimentState(BaseModel):
    """Canonical representation of an experiment in Redis.

    The fields map 1-to-1 to the hash stored under ``experiment:{id}``.
    All numeric values are kept as proper numbers in Python; the
    ``RedisBroker`` is responsible for serialising to strings.
    """

    id: str
    status: str
    config: Dict[str, Any]
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    current_step: int = 0
    total_steps: int = 0
    metrics: Dict[str, Any] = Field(default_factory=dict)
    message: Optional[str] = None
    last_updated: Optional[float] = None

    @field_validator("status")
    @classmethod
    def _lowercase_status(cls, v: str) -> str:  # noqa: D401 (simple)
        return v.lower()

    def to_flat_dict(self) -> Dict[str, str]:
        """Convert to a flat str→str mapping suitable for ``redis.hset``."""
        out: Dict[str, str] = {}
        for field, value in self.model_dump().items():
            if isinstance(value, dict):
                out[field] = json.dumps(value)
            elif value is None:
                out[field] = "None"
            else:
                out[field] = str(value)
        return out

    @classmethod
    def from_redis_hash(cls, data: Dict[str, str]) -> "ExperimentState":
        """Instantiate from a hash returned by Redis."""
        d: Dict[str, Any] = {}
        for k, v in data.items():
            if k in {"config", "metrics"}:
                try:
                    d[k] = json.loads(v)
                except Exception:
                    d[k] = {}
            elif k in {"start_time", "end_time", "duration", "last_updated"}:
                d[k] = float(v) if v != "None" else None
            elif k in {"current_step", "total_steps"}:
                d[k] = int(v)
            else:
                d[k] = v
        return cls(**d) 