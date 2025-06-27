# flake8: noqa
"""Async Redis-backed implementation of :class:`src.messaging.base.MessageBroker`."""

from __future__ import annotations

import asyncio
import json
import traceback
from typing import Any, AsyncIterator, Dict, List, Optional

from redis import asyncio as redis  # type: ignore

from src.messaging.base import MessageBroker, Message
from src.messaging.config import RedisConfig
from src.types.experiments import Experiment, ExperimentState, TrainingStatus, ExperimentConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

class RedisBroker(MessageBroker):
    """Redis-based message broker using pub/sub and hash storage."""

    def __init__(self, config: RedisConfig):
        self.config = config
        self._redis: Optional[redis.Redis] = None  # type: ignore[attr-defined]
        self._pubsub: Optional[redis.client.PubSub] = None  # type: ignore[attr-defined]
        self._subscriptions: Dict[str, asyncio.Task] = {}

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------
    async def connect(self) -> None:  # noqa: D401 (simple)
        self._redis = await redis.from_url(
            f"redis://{self.config.host}:{self.config.port}/{self.config.db}",
            password=self.config.password,
            socket_timeout=self.config.socket_timeout,
            socket_connect_timeout=self.config.socket_connect_timeout,
            decode_responses=True,
        )
        self._pubsub = self._redis.pubsub()
        logger.info("Connected to Redis at %s:%s", self.config.host, self.config.port)

    async def disconnect(self) -> None:
        if self._pubsub is not None:
            await self._pubsub.close()
        if self._redis is not None:
            await self._redis.close()
        self._pubsub = None
        self._redis = None

    async def close(self) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Pub/Sub
    # ------------------------------------------------------------------
    async def publish(self, topic: str, data: Dict[str, Any]) -> None:
        if self._redis is None:
            await self.connect()
        assert self._redis is not None
        await self._redis.publish(topic, json.dumps(data))

    async def subscribe(self, pattern: str) -> AsyncIterator[Message]:
        if self._redis is None or self._pubsub is None:
            await self.connect()
        assert self._pubsub is not None
        await self._pubsub.psubscribe(pattern)  # type: ignore[attr-defined]
        try:
            while True:
                raw = await self._pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if raw is None:
                    await asyncio.sleep(0.1)
                    continue
                try:
                    data = json.loads(raw["data"])
                    yield Message(topic=raw["channel"], data=data, timestamp=None)
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning("Failed to parse redis message: %s", exc)
        finally:
            await self._pubsub.punsubscribe(pattern)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Experiment storage identical in behaviour to previous file
    # ------------------------------------------------------------------
    async def store_experiment(self, exp: Experiment) -> None:
        if self._redis is None:
            await self.connect()
        assert self._redis is not None
        key = f"experiment:{exp.id}"
        mapping = exp.to_redis_hash()
        await self._redis.hset(key, mapping=mapping)
        await self._redis.sadd("active_experiments", exp.id)
        status_value = exp.state.status.value
        if status_value in {TrainingStatus.COMPLETED.value, TrainingStatus.FAILED.value, TrainingStatus.CANCELLED.value}:
            await self._redis.expire(key, 86400)

    async def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        if self._redis is None:
            await self.connect()
        assert self._redis is not None
        key = f"experiment:{experiment_id}"
        data = await self._redis.hgetall(key)
        if not data:
            return None
        return Experiment.from_redis_hash(data)

    async def update_experiment(self, experiment_id: str, updates: Dict[str, Any]) -> None:
        if self._redis is None:
            await self.connect()
        assert self._redis is not None
        key = f"experiment:{experiment_id}"
        exists = await self._redis.exists(key)
        if not exists:
            logger.error("Experiment %s not found for update", experiment_id)
            return
        flattened: Dict[str, str] = {}
        for field, value in updates.items():
            if value is None:
                # Explicitly delete keys that should become null/absent to
                # keep the hash clean – this prevents stray "None" strings.
                await self._redis.hdel(key, field)
                continue
            if isinstance(value, dict):
                flattened[field] = json.dumps(value, separators=(",", ":"))
            elif field == "status" and hasattr(value, "value"):
                flattened[field] = value.value  # enum → str
            else:
                flattened[field] = str(value)
        if flattened:
            await self._redis.hset(key, mapping=flattened)

    async def list_experiments(self, status_filter: Optional[Any] = None) -> List[Experiment]:
        if self._redis is None:
            await self.connect()
        assert self._redis is not None
        experiment_ids = await self._redis.smembers("active_experiments")
        experiments: List[Experiment] = []
        for exp_id in experiment_ids:
            exp = await self.get_experiment(exp_id)
            if exp:
                if status_filter is None:
                    experiments.append(exp)
                else:
                    # Accept str or Iterable[str]
                    if isinstance(status_filter, str):
                        if exp.status.value == status_filter:
                            experiments.append(exp)
                    else:
                        if exp.status.value in status_filter:
                            experiments.append(exp)
        experiments.sort(key=lambda x: x.state.start_time or 0, reverse=True)
        return experiments

    async def remove_experiment(self, experiment_id: str) -> None:
        if self._redis is None:
            await self.connect()
        assert self._redis is not None
        await self._redis.srem("active_experiments", experiment_id)
        await self._redis.delete(f"experiment:{experiment_id}")

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    async def health_check(self) -> bool:
        try:
            if self._redis is None:
                await self.connect()
            assert self._redis is not None
            pong = await self._redis.ping()
            return pong is True
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:  # noqa: D401 (simple)
        return {
            "broker_type": "redis",
            "subscriptions": len(self._subscriptions),
            "connected": self._redis is not None,
        } 