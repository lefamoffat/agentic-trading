"""Event publishing for training experiments."""

from typing import Dict, Optional

from src.messaging import (
    TrainingStatus,
    create_status_event,
    create_progress_event,
    create_metrics_event,
    create_log_event,
    create_error_event
)


class EventPublisher:
    """Handles publishing of training events."""
    
    def __init__(self, message_broker):
        """Initialize event publisher.
        
        Args:
            message_broker: Message broker instance for publishing events
        """
        self.message_broker = message_broker
    
    async def publish_status_update(
        self,
        experiment_id: str,
        status: TrainingStatus,
        message: Optional[str] = None
    ) -> None:
        """Publish status update event."""
        if self.message_broker:
            event = create_status_event(experiment_id, status, message)
            await self.message_broker.publish(
                event.event_type.value,
                event.to_message().data
            )
            
    async def publish_progress_update(
        self,
        experiment_id: str,
        current_step: int,
        total_steps: int,
        epoch: Optional[int] = None
    ) -> None:
        """Publish progress update event."""
        if self.message_broker:
            event = create_progress_event(experiment_id, current_step, total_steps, epoch)
            await self.message_broker.publish(
                event.event_type.value,
                event.to_message().data
            )
            
    async def publish_metrics_update(
        self,
        experiment_id: str,
        metrics: Dict[str, float]
    ) -> None:
        """Publish metrics update event."""
        if self.message_broker:
            event = create_metrics_event(experiment_id, metrics)
            await self.message_broker.publish(
                event.event_type.value,
                event.to_message().data
            )
            
    async def publish_log_update(
        self,
        experiment_id: str,
        level: str,
        message: str,
        source: Optional[str] = None
    ) -> None:
        """Publish log update event."""
        if self.message_broker:
            event = create_log_event(experiment_id, level, message, source)
            await self.message_broker.publish(
                event.event_type.value,
                event.to_message().data
            )
            
    async def publish_error(
        self,
        experiment_id: str,
        error_type: str,
        error_message: str,
        traceback_str: Optional[str] = None
    ) -> None:
        """Publish error event."""
        if self.message_broker:
            event = create_error_event(experiment_id, error_type, error_message, traceback_str)
            await self.message_broker.publish(
                event.event_type.value,
                event.to_message().data
            ) 