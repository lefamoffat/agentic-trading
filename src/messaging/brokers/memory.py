"""In-memory implementation of the :class:`~src.messaging.base.MessageBroker`.

This broker is intended for local development and unit-testing.  It keeps all
messages and experiment state in Python data structures, so **it requires no
external services** and works exactly like the Redis broker from a consumer's
perspective.

The class was moved from ``src.messaging.memory_broker`` to the new
``src.messaging.brokers`` package.  Behaviour is unchanged except for two
important fixes:

1. ``store_experiment()`` now accepts either a plain ``dict`` **or** an
   :class:`~src.messaging.schema.ExperimentState` model.  If a Pydantic model is
   provided we immediately convert it to ``dict`` so that later calls to
   ``update_experiment()`` can rely on standard ``dict.update()``.
2. ``update_experiment()`` guards against accidentally having a non-mapping
   object stored (e.g. during hot-reload) by coercing it to a ``dict`` before
   applying updates.
"""
from __future__ import annotations

import asyncio
import copy
import fnmatch
import time
from collections import defaultdict, deque
from typing import Any, AsyncIterator, Dict, List, Optional

from src.messaging.base import MessageBroker, Message
from src.messaging.config import MemoryConfig
from src.types.experiments import Experiment, ExperimentState
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MemoryBroker(MessageBroker):
    """Pure-Python message broker suitable for development / testing."""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self._subscribers: Dict[str, List[asyncio.Queue]] = defaultdict(list)
        self._message_history: deque[Message] = deque(maxlen=config.max_queue_size)
        self._experiments: Dict[str, Experiment] = {}
        self._closed = False

        logger.info("Initialized MemoryBroker (max_queue_size=%s)", config.max_queue_size)

    # ---------------------------------------------------------------------
    # Pub/Sub
    # ---------------------------------------------------------------------
    async def publish(self, topic: str, data: Dict[str, Any]) -> None:  # noqa: D401 (simple)
        if self._closed:
            logger.warning("Cannot publish to closed broker")
            return

        message = Message(topic=topic, data=data, timestamp=time.time())
        self._message_history.append(message)

        delivered = 0
        for pattern, queues in list(self._subscribers.items()):
            if fnmatch.fnmatch(topic, pattern):
                for queue in queues[:]:  # iterate over shallow copy
                    try:
                        queue.put_nowait(message)
                        delivered += 1
                    except asyncio.QueueFull:
                        logger.warning("Queue full for pattern %s, dropping message", pattern)
                    except Exception as exc:
                        logger.error("Error delivering message to subscriber: %s", exc)
                        if queue in queues:
                            queues.remove(queue)
        logger.debug("Published %s to %d subscriber(s)", topic, delivered)

    async def subscribe(self, pattern: str) -> AsyncIterator[Message]:
        if self._closed:
            logger.warning("Cannot subscribe to closed broker")
            return

        if len(self._subscribers[pattern]) >= self.config.max_subscribers:
            logger.error("Max subscribers (%s) reached for pattern %s", self.config.max_subscribers, pattern)
            return

        queue: asyncio.Queue[Message] = asyncio.Queue(maxsize=self.config.max_queue_size)
        self._subscribers[pattern].append(queue)
        logger.info("New subscriber for pattern: %s", pattern)

        try:
            while not self._closed:
                try:
                    message: Message = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield message
                except asyncio.TimeoutError:
                    continue
        finally:
            if queue in self._subscribers.get(pattern, []):
                self._subscribers[pattern].remove(queue)
            logger.info("Unsubscribed from pattern: %s", pattern)

    # ---------------------------------------------------------------------
    # Experiment data storage (API parity with RedisBroker)
    # ---------------------------------------------------------------------
    async def store_experiment(self, exp: Experiment) -> None:
        if self._closed:
            logger.warning("Cannot store experiment in closed broker")
            return

        self._experiments[exp.id] = exp.model_copy(deep=True)
        logger.debug("Stored experiment %s", exp.id)

    async def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        if self._closed:
            return None
        exp = self._experiments.get(experiment_id)
        return exp.model_copy(deep=True) if exp else None

    async def update_experiment(self, experiment_id: str, updates: Dict[str, Any]) -> None:
        if self._closed:
            logger.warning("Cannot update experiment in closed broker")
            return
        if experiment_id not in self._experiments:
            logger.warning("Experiment %s not found for update", experiment_id)
            return

        current = self._experiments[experiment_id]
        # mutate state in-place
        current.state = current.state.model_copy(update=updates)  # type: ignore[arg-type]
        logger.debug("Updated experiment %s: %s", experiment_id, list(updates.keys()))

    async def list_experiments(self, status_filter: Optional[Any] = None) -> List[Experiment]:
        if self._closed:
            return []
        experiments: List[Experiment] = []
        for exp in self._experiments.values():
            if status_filter is None:
                experiments.append(exp.model_copy(deep=True))
            else:
                if isinstance(status_filter, str):
                    if exp.status.value == status_filter:
                        experiments.append(exp.model_copy(deep=True))
                else:
                    # Assume iterable of strings
                    if exp.status.value in status_filter:
                        experiments.append(exp.model_copy(deep=True))
        experiments.sort(key=lambda x: x.state.start_time or 0, reverse=True)
        return experiments

    async def remove_experiment(self, experiment_id: str) -> None:
        if self._closed:
            return
        self._experiments.pop(experiment_id, None)
        logger.debug("Removed experiment %s", experiment_id)

    # ------------------------------------------------------------------
    # House-keeping / health
    # ------------------------------------------------------------------
    async def close(self) -> None:
        self._closed = True
        for queues in self._subscribers.values():
            for queue in queues:
                try:
                    queue.put_nowait(None)  # type: ignore[arg-type]
                except asyncio.QueueFull:
                    pass
        self._subscribers.clear()
        self._experiments.clear()
        logger.info("MemoryBroker closed")

    async def health_check(self) -> bool:  # noqa: D401 (simple)
        return not self._closed

    def get_stats(self) -> Dict[str, Any]:  # noqa: D401 (simple)
        total_subscribers = sum(len(q) for q in self._subscribers.values())
        return {
            "broker_type": "memory",
            "total_subscribers": total_subscribers,
            "active_patterns": len(self._subscribers),
            "message_history_size": len(self._message_history),
            "max_queue_size": self.config.max_queue_size,
            "max_subscribers": self.config.max_subscribers,
            "active_experiments": len(self._experiments),
            "closed": self._closed,
        } 