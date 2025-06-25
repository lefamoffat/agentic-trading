"""Event definitions and utilities for training status updates."""

from dataclasses import dataclass
from typing import Any, Dict, Optional
from enum import Enum
import time

from src.messaging.base import Message

class TrainingStatus(Enum):
    """Training status enumeration."""
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class EventType(Enum):
    """Event type enumeration."""
    STATUS_UPDATE = "training.status"
    PROGRESS_UPDATE = "training.progress"
    METRICS_UPDATE = "training.metrics"
    LOG_UPDATE = "training.logs"
    ERROR = "training.error"

@dataclass
class TrainingEvent:
    """Base training event."""
    experiment_id: str
    timestamp: float
    event_type: EventType
    data: Dict[str, Any]
    
    def to_message(self) -> Message:
        """Convert to Message object."""
        return Message(
            topic=self.event_type.value,
            data={
                "experiment_id": self.experiment_id,
                "timestamp": self.timestamp,
                **self.data
            },
            timestamp=self.timestamp
        )

@dataclass
class StatusUpdateEvent(TrainingEvent):
    """Training status update event."""
    status: TrainingStatus
    message: Optional[str] = None
    
    def __post_init__(self):
        self.event_type = EventType.STATUS_UPDATE
        self.data = {
            "status": self.status.value,
            "message": self.message
        }

@dataclass
class ProgressUpdateEvent(TrainingEvent):
    """Training progress update event."""
    current_step: int
    total_steps: int
    epoch: Optional[int] = None
    
    def __post_init__(self):
        self.event_type = EventType.PROGRESS_UPDATE
        self.data = {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress": self.current_step / self.total_steps if self.total_steps > 0 else 0.0,
            "epoch": self.epoch
        }

@dataclass
class MetricsUpdateEvent(TrainingEvent):
    """Training metrics update event."""
    metrics: Dict[str, float]
    
    def __post_init__(self):
        self.event_type = EventType.METRICS_UPDATE
        self.data = {
            "metrics": self.metrics
        }

@dataclass
class LogUpdateEvent(TrainingEvent):
    """Training log update event."""
    level: str
    message: str
    source: Optional[str] = None
    
    def __post_init__(self):
        self.event_type = EventType.LOG_UPDATE
        self.data = {
            "level": self.level,
            "message": self.message,
            "source": self.source
        }

@dataclass
class ErrorEvent(TrainingEvent):
    """Training error event."""
    error_type: str
    error_message: str
    traceback: Optional[str] = None
    
    def __post_init__(self):
        self.event_type = EventType.ERROR
        self.data = {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "traceback": self.traceback
        }

# Helper functions for creating events

def create_status_event(
    experiment_id: str,
    status: TrainingStatus,
    message: Optional[str] = None
) -> StatusUpdateEvent:
    """Create a status update event."""
    return StatusUpdateEvent(
        experiment_id=experiment_id,
        timestamp=time.time(),
        status=status,
        message=message,
        event_type=EventType.STATUS_UPDATE,
        data={}
    )

def create_progress_event(
    experiment_id: str,
    current_step: int,
    total_steps: int,
    epoch: Optional[int] = None
) -> ProgressUpdateEvent:
    """Create a progress update event."""
    return ProgressUpdateEvent(
        experiment_id=experiment_id,
        timestamp=time.time(),
        current_step=current_step,
        total_steps=total_steps,
        epoch=epoch,
        event_type=EventType.PROGRESS_UPDATE,
        data={}
    )

def create_metrics_event(
    experiment_id: str,
    metrics: Dict[str, float]
) -> MetricsUpdateEvent:
    """Create a metrics update event."""
    return MetricsUpdateEvent(
        experiment_id=experiment_id,
        timestamp=time.time(),
        metrics=metrics,
        event_type=EventType.METRICS_UPDATE,
        data={}
    )

def create_log_event(
    experiment_id: str,
    level: str,
    message: str,
    source: Optional[str] = None
) -> LogUpdateEvent:
    """Create a log update event."""
    return LogUpdateEvent(
        experiment_id=experiment_id,
        timestamp=time.time(),
        level=level,
        message=message,
        source=source,
        event_type=EventType.LOG_UPDATE,
        data={}
    )

def create_error_event(
    experiment_id: str,
    error_type: str,
    error_message: str,
    traceback: Optional[str] = None
) -> ErrorEvent:
    """Create an error event."""
    return ErrorEvent(
        experiment_id=experiment_id,
        timestamp=time.time(),
        error_type=error_type,
        error_message=error_message,
        traceback=traceback,
        event_type=EventType.ERROR,
        data={}
    ) 