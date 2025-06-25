"""Real-time messaging system for training events and dashboard updates.

This module provides pub/sub messaging capabilities for real-time communication
between training processes, dashboard, and CLI components.

Example Usage:
    # Publishing events
    broker = get_message_broker()
    await broker.publish("training.status", {
        "experiment_id": "exp_123",
        "status": "running", 
        "progress": 0.45
    })
    
    # Subscribing to events
    async for event in broker.subscribe("training.*"):
        print(f"Received: {event.topic} - {event.data}")
"""

from src.messaging.base import MessageBroker, Message
from src.messaging.events import (
    TrainingStatus,
    EventType,
    TrainingEvent,
    StatusUpdateEvent,
    ProgressUpdateEvent,
    MetricsUpdateEvent,
    LogUpdateEvent,
    ErrorEvent,
    create_status_event,
    create_progress_event,
    create_metrics_event,
    create_log_event,
    create_error_event
)

__all__ = [
    "MessageBroker",
    "Message",
    "TrainingStatus",
    "EventType",
    "TrainingEvent",
    "StatusUpdateEvent", 
    "ProgressUpdateEvent",
    "MetricsUpdateEvent",
    "LogUpdateEvent",
    "ErrorEvent",
    "create_status_event",
    "create_progress_event",
    "create_metrics_event",
    "create_log_event",
    "create_error_event"
] 