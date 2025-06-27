"""Training-specific messaging channel."""

import time
from typing import Any, AsyncIterator, Dict, Optional, List
import traceback

from src.messaging.channels.base import BaseChannel
from src.messaging.base import MessageBroker, Message
from src.messaging.events import (
    TrainingStatus,
    create_status_event,
    create_progress_event,
    create_metrics_event,
    create_log_event,
    create_error_event
)
from src.utils.logger import get_logger
from src.types.experiments import Experiment, ExperimentState, ExperimentConfig

logger = get_logger(__name__)


class TrainingChannel(BaseChannel):
    """Channel for training-related messaging and data storage."""
    
    def __init__(self, broker: MessageBroker):
        """Initialize training channel."""
        super().__init__(broker, "training")
    
    # === EXPERIMENT LIFECYCLE ===
    
    async def create_experiment(self, experiment_id: str, config: Dict[str, Any]) -> None:
        """
        Create a new experiment and store its initial state.
        
        Args:
            experiment_id: Unique experiment identifier
            config: Experiment configuration
        """
        cfg_model = ExperimentConfig(**config)
        state_model = ExperimentState(
            status=TrainingStatus.STARTING,
            start_time=time.time(),
            total_steps=cfg_model.timesteps,
        )

        exp = Experiment(id=experiment_id, config=cfg_model, state=state_model)

        await self.broker.store_experiment(exp)
        
        # Publish creation event
        await self.publish("experiment.created", {
            "experiment_id": experiment_id,
            "config": config,
            "timestamp": time.time()
        })
        
        logger.info(f"Created experiment {experiment_id}")
    
    async def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment data."""
        return await self.broker.get_experiment(experiment_id)
    
    async def list_experiments(self, status_filter: Optional[Any] = None) -> List[Experiment]:
        """List experiments, optionally filtered by status."""
        return await self.broker.list_experiments(status_filter)
    
    async def remove_experiment(self, experiment_id: str) -> None:
        """Remove experiment from active storage."""
        await self.broker.remove_experiment(experiment_id)
        
        # Publish removal event
        await self.publish("experiment.removed", {
            "experiment_id": experiment_id,
            "timestamp": time.time()
        })
    
    # === STATUS UPDATES ===
    
    async def publish_status(self, experiment_id: str, status: TrainingStatus, message: Optional[str] = None) -> None:
        """
        Publish training status update.
        
        Args:
            experiment_id: Experiment identifier
            status: New status
            message: Optional status message
        """
        terminal_statuses = {
            TrainingStatus.COMPLETED.value,
            TrainingStatus.FAILED.value,
            TrainingStatus.CANCELLED.value
        }
        
        updates = {
            "status": status.value,
            "last_updated": time.time()
        }
        
        if message:
            updates["message"] = message
            
        logger.info(f"[STATUS] Publishing status update for {experiment_id}: {status.value} ({message})")
        
        if status.value in terminal_statuses:
            logger.info(f"[STATUS] Terminal status detected for {experiment_id}")
            updates["end_time"] = time.time()
            # Calculate duration if we have start_time
            experiment = await self.get_experiment(experiment_id)
            if experiment and experiment.state.start_time:
                updates["duration"] = updates["end_time"] - experiment.state.start_time
                logger.info(f"[STATUS] Updated duration: {updates['duration']} seconds")
        
        try:
            # First update experiment status in storage
            logger.info(f"[STATUS] Updating experiment in broker: {updates}")
            await self.broker.update_experiment(experiment_id, updates)
            logger.info(f"[STATUS] Successfully updated experiment in broker")
            
            # Then publish status event
            event = create_status_event(experiment_id, status, message)
            message_obj = event.to_message()
            logger.info(f"[STATUS] Publishing status event: {message_obj.data}")
            await self.publish("status", message_obj.data)
            logger.info(f"[STATUS] Successfully published status event")
            
        except Exception as e:
            logger.error(f"[STATUS] Failed to update status: {e}")
            logger.error(f"[STATUS] Error type: {type(e).__name__}")
            logger.error(f"[STATUS] Error traceback:\n{traceback.format_exc()}")
            # Re-raise to ensure error is propagated
            raise
    
    async def publish_progress(self, experiment_id: str, current_step: int, total_steps: int, step: int = None) -> None:
        """
        Publish training progress update.
        
        Args:
            experiment_id: Experiment identifier
            current_step: Current training step
            total_steps: Total training steps
            step: Optional current step
        """
        # Guarantee invariant current_step ≤ total_steps -------------------
        if current_step > total_steps:
            total_steps = current_step  # training overshot original target

        logger.debug(
            "[CHANNEL] publish_progress: %s / %s (exp %s)", current_step, total_steps, experiment_id
        )

        await self.broker.update_experiment(
            experiment_id,
            {
                "current_step": current_step,
                "total_steps": total_steps,
                "last_updated": time.time(),
            },
        )
        
        # Create and publish event with potentially bumped total_steps
        event = create_progress_event(experiment_id, current_step, total_steps, step)
        message_obj = event.to_message()
        await self.publish("progress", message_obj.data)
        
        logger.debug(f"Published progress for {experiment_id}: {current_step}/{total_steps}")
    
    async def publish_metrics(self, experiment_id: str, metrics: Dict[str, float]) -> None:
        """
        Publish training metrics update.
        
        Args:
            experiment_id: Experiment identifier
            metrics: Training metrics dictionary
        """
        # Get current experiment to merge metrics
        experiment = await self.get_experiment(experiment_id)
        if experiment:
            current_metrics = experiment.state.metrics.copy()
            current_metrics.update(metrics)
            
            await self.broker.update_experiment(experiment_id, {
                "metrics": current_metrics,
                "last_updated": time.time()
            })
        
        # Create and publish event
        event = create_metrics_event(experiment_id, metrics)
        message_obj = event.to_message()
        await self.publish("metrics", message_obj.data)
        
        logger.debug(f"Published metrics for {experiment_id}: {list(metrics.keys())}")
    
    async def publish_log(self, experiment_id: str, level: str, message: str, source: Optional[str] = None) -> None:
        """
        Publish training log message.
        
        Args:
            experiment_id: Experiment identifier
            level: Log level (info, warning, error, etc.)
            message: Log message
            source: Optional log source
        """
        event = create_log_event(experiment_id, level, message, source)
        message_obj = event.to_message()
        await self.publish("logs", message_obj.data)
    
    async def publish_error(self, experiment_id: str, error_type: str, error_message: str, traceback: Optional[str] = None) -> None:
        """
        Publish training error.
        
        Args:
            experiment_id: Experiment identifier
            error_type: Type of error
            error_message: Error message
            traceback: Optional error traceback
        """
        # Update experiment status to failed
        await self.broker.update_experiment(experiment_id, {
            "status": TrainingStatus.FAILED.value,
            "error": {
                "type": error_type,
                "message": error_message,
                "traceback": traceback
            },
            "end_time": time.time(),
            "last_updated": time.time()
        })
        
        # Create and publish event
        event = create_error_event(experiment_id, error_type, error_message, traceback)
        message_obj = event.to_message()
        await self.publish("error", message_obj.data)
        
        logger.error(f"Published error for {experiment_id}: {error_type} - {error_message}")
    
    # === SUBSCRIPTIONS ===
    
    async def subscribe_to_experiment(self, experiment_id: str) -> AsyncIterator[Message]:
        """
        Subscribe to all events for a specific experiment.
        
        Args:
            experiment_id: Experiment identifier
            
        Yields:
            Messages for the specified experiment
        """
        async for message in self.subscribe("*"):
            # Filter messages for this experiment
            if message.data.get("experiment_id") == experiment_id:
                yield message
    
    async def subscribe_to_status_updates(self) -> AsyncIterator[Message]:
        """Subscribe to all training status updates."""
        async for message in self.subscribe("status"):
            yield message
    
    async def subscribe_to_progress_updates(self) -> AsyncIterator[Message]:
        """Subscribe to all training progress updates."""
        async for message in self.subscribe("progress"):
            yield message
    
    async def subscribe_to_metrics_updates(self) -> AsyncIterator[Message]:
        """Subscribe to all training metrics updates."""
        async for message in self.subscribe("metrics"):
            yield message 