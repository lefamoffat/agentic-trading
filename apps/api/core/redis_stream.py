from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Set

import redis.asyncio as aioredis

from src.utils.logger import get_logger

logger = get_logger(__name__)

_PATTERN = "training.*"


class StreamManager:
    """Subscribe to Redis pub/sub channel and fan-out messages to queues."""

    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or "redis://localhost:6379/0"
        self._redis: aioredis.Redis | None = None
        self._pubsub: aioredis.client.PubSub | None = None
        self._queues: Set[asyncio.Queue] = set()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task and not self._task.done():
            return  # already running
        self._redis = aioredis.from_url(self.redis_url, decode_responses=True)
        self._pubsub = self._redis.pubsub()
        await self._pubsub.psubscribe(_PATTERN)
        self._task = asyncio.create_task(self._relay(), name="redis-relay")
        logger.info("StreamManager: psubscribed to pattern %s", _PATTERN)

    async def _relay(self) -> None:
        assert self._pubsub is not None
        async for message in self._pubsub.listen():
            if message["type"] not in ("message", "pmessage"):
                continue
            try:
                payload: Dict[str, Any] = json.loads(message["data"])
            except json.JSONDecodeError:
                logger.warning("Invalid JSON via Redis pubsub pattern %s", _PATTERN)
                continue
            # Fan-out to listeners
            dead: Set[asyncio.Queue] = set()
            for queue in self._queues:
                try:
                    queue.put_nowait(payload)
                except asyncio.QueueFull:
                    logger.debug("Listener queue full; dropping message")
            # Remove any cancelled queues
            self._queues.difference_update(dead)

    def register(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._queues.add(q)
        return q

    def unregister(self, queue: asyncio.Queue) -> None:
        self._queues.discard(queue) 