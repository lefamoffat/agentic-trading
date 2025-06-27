"""Broker-agnostic fan-out relay used by the API WebSocket layer.

A single instance subscribes to the configured :class:`MessageBroker` and fans
messages out to multiple `asyncio.Queue` listeners.  Works for any concrete
broker (memory or redis) because it relies solely on the abstract interface.
"""
from __future__ import annotations

import asyncio
import contextlib
from typing import Set

from src.messaging.base import MessageBroker, Message
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BrokerRelay:  # noqa: WPS110 (simple container class)
    """Subscribe once to the broker and relay matching messages to listeners."""

    def __init__(self, broker: MessageBroker, pattern: str = "training.*") -> None:
        self._broker = broker
        self._pattern = pattern
        self._queues: Set[asyncio.Queue] = set()
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="broker-relay")
        logger.info("BrokerRelay started for pattern %s", self._pattern)

    async def _run(self) -> None:  # noqa: WPS231
        try:
            async for msg in self._broker.subscribe(self._pattern):
                await self._fan_out(msg)
        except asyncio.CancelledError:  # propagate task cancellation
            pass
        except Exception as exc:  # noqa: BLE001
            logger.error("BrokerRelay stopped due to error: %s", exc)

    async def _fan_out(self, msg: Message) -> None:
        dead: Set[asyncio.Queue] = set()
        for listener_queue in self._queues:
            try:
                # WebSocket clients expect a wrapper object with the topic so
                # they can multiplex different sub-streams.  Relay therefore
                # normalises every broker message to the canonical structure
                #   {"topic": "training.progress", "data": {...}}
                payload = {
                    "topic": msg.topic,
                    "data": msg.data,
                }
                listener_queue.put_nowait(payload)
            except asyncio.QueueFull:
                logger.debug("Listener queue full – dropping message")
            except RuntimeError:  # queue closed
                dead.add(listener_queue)
        self._queues.difference_update(dead)

    # ------------------------------------------------------------------
    def register(self) -> asyncio.Queue:
        listener_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._queues.add(listener_queue)
        return listener_queue

    def unregister(self, listener_queue: asyncio.Queue) -> None:
        self._queues.discard(listener_queue)

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._queues.clear()
        logger.info("BrokerRelay stopped") 