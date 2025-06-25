from __future__ import annotations

import time
from typing import Any
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback

from src.utils.logger import get_logger
from ..metric_aggregator import MetricAggregator
from ..progress_dispatcher import ProgressDispatcher

logger = get_logger(__name__)


class SB3TrainingCallback(BaseCallback):
    """Thin Stable-Baselines3 callback that leans on helper classes."""

    def __init__(
        self,
        experiment_id: str,
        training_channel,
        event_publisher,
        loop,
        total_timesteps: int,
        ml_tracker=None,
        ml_run=None,
    ) -> None:
        super().__init__()
        self.experiment_id = experiment_id
        self.total_timesteps = total_timesteps
        self._progress_interval = 0.5  # seconds
        self._metrics_interval = 1.0
        self._last_progress_time = 0.0
        self._last_metrics_time = 0.0

        self.aggregator = MetricAggregator()
        self.dispatcher = ProgressDispatcher(experiment_id, training_channel, event_publisher, loop, ml_tracker, ml_run)

    # ------------------------------------------------------------------
    # SB3 hooks
    # ------------------------------------------------------------------

    def _on_step(self) -> bool:  # noqa: D401 (SB3 signature)
        now = time.time()
        step = self.num_timesteps

        # update aggregator from env info
        info = (self.locals.get("infos") or [{}])[-1]
        # Retrieve reward from locals if provided by SB3
        reward_value = 0.0
        if "rewards" in self.locals:
            rew_obj = self.locals["rewards"]
            # Handle scalar, NumPy array, or list uniformly
            if isinstance(rew_obj, (float, int)):
                reward_value = float(rew_obj)
            elif isinstance(rew_obj, np.ndarray):
                reward_value = float(rew_obj.mean())
            elif isinstance(rew_obj, (list, tuple)) and rew_obj:
                reward_value = float(rew_obj[-1])
        self.aggregator.update(info, reward_value)

        # progress
        if now - self._last_progress_time >= self._progress_interval:
            self._last_progress_time = now
            self.dispatcher.publish_progress(step, self.total_timesteps)

        # metrics
        if now - self._last_metrics_time >= self._metrics_interval:
            self._last_metrics_time = now
            metrics = self.aggregator.summary()
            if metrics:
                metrics["step"] = step
                self.dispatcher.publish_metrics(metrics)

        return True 