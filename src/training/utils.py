from __future__ import annotations

"""Utility helpers for the training sub-package."""

import asyncio
import functools
import traceback
from typing import Callable, TypeVar, Awaitable, Any

from src.utils.logger import get_logger
from src.messaging import TrainingStatus

logger = get_logger(__name__)

T = TypeVar("T")


def async_guard(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Decorator that catches exceptions in background tasks and publishes them.

    The wrapped coroutine **must** be a method on a class that provides the
    attributes ``training_channel`` and ``event_publisher``.
    """

    @functools.wraps(func)
    async def wrapper(self: Any, *args, **kwargs):  # type: ignore[override]
        try:
            return await func(self, *args, **kwargs)
        except Exception as exc:
            experiment_id = args[0] if args else "unknown"
            logger.error(
                f"[async_guard] Exception in {func.__qualname__} for exp {experiment_id}: {exc}\n"
                f"{traceback.format_exc()}"
            )
            try:
                # publish failure to redis/message bus
                await self.training_channel.publish_status(experiment_id, TrainingStatus.FAILED, str(exc))
                await self.event_publisher.publish_error(experiment_id, type(exc).__name__, str(exc), traceback.format_exc())
            except Exception as pub_err:  # noqa: BLE001
                logger.error(f"[async_guard] Failed to publish error: {pub_err}")
            raise  # let upstream handler decide further

    return wrapper 