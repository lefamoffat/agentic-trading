from __future__ import annotations

import asyncio
from typing import Dict, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ProgressDispatcher:
    """Sends progress & metrics to TrainingChannel and EventPublisher."""

    def __init__(self, experiment_id: str, training_channel, event_publisher, loop, ml_tracker=None, ml_run=None):
        self.experiment_id = experiment_id
        self.training_channel = training_channel
        self.event_publisher = event_publisher
        self.loop = loop
        self.ml_tracker = ml_tracker
        self.ml_run = ml_run

    def publish_progress(self, current_step: int, total_steps: int) -> None:
        coro = self.training_channel.publish_progress(self.experiment_id, current_step, total_steps)
        asyncio.run_coroutine_threadsafe(coro, self.loop)

    def publish_metrics(self, metrics: Dict[str, Any]) -> None:
        coro = self.training_channel.publish_metrics(self.experiment_id, metrics)
        asyncio.run_coroutine_threadsafe(coro, self.loop)
        # also send via event publisher
        if self.event_publisher:
            coro2 = self.event_publisher.publish_metrics_update(self.experiment_id, metrics)
            asyncio.run_coroutine_threadsafe(coro2, self.loop)

        # ML tracker logging (fire and forget)
        if self.ml_tracker and self.ml_run:
            coro3 = self.ml_tracker.log_metrics(self.ml_run.id, metrics, metrics.get("step", 0))
            asyncio.run_coroutine_threadsafe(coro3, self.loop) 