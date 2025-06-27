"""Lazy loader for concrete message broker implementations.

Imports are deferred until a specific broker is requested to avoid loading
unnecessary dependencies in environments that don't need them.
"""

from __future__ import annotations

from typing import Type

from src.messaging.base import MessageBroker


def get_broker_cls(broker_type: str) -> Type[MessageBroker]:
    """Return the broker class matching *broker_type*.

    Imports are performed lazily so optional dependencies are loaded only when
    the corresponding broker type is actually requested.
    """

    norm = broker_type.lower()
    if norm == "memory":
        from .memory import MemoryBroker  # local import (no heavy deps)

        return MemoryBroker  # type: ignore[return-value]

    if norm == "redis":
        from .redis import RedisBroker  # local import, pulls aioredis

        return RedisBroker  # type: ignore[return-value]

    raise ValueError(f"Unknown broker type: {broker_type}")


__all__ = ["get_broker_cls"] 