"""Messaging channels module with factory functions."""

from typing import Optional

from src.messaging.factory import get_message_broker
from src.messaging.channels.training import TrainingChannel

# Global instance for singleton pattern
_training_channel_instance: Optional[TrainingChannel] = None

async def get_training_channel() -> TrainingChannel:
    """
    Get or create the global training channel instance.
    
    Returns:
        TrainingChannel instance connected to the message broker
    """
    global _training_channel_instance
    
    if _training_channel_instance is None:
        message_broker = get_message_broker()
        _training_channel_instance = TrainingChannel(message_broker)
    
    return _training_channel_instance

def reset_training_channel():
    """Reset the global training channel instance (for testing)."""
    global _training_channel_instance
    _training_channel_instance = None

__all__ = [
    "get_training_channel",
    "reset_training_channel",
    "TrainingChannel"
] 