"""Abstract base class for messaging channels."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Optional, List

from src.messaging.base import MessageBroker, Message


class BaseChannel(ABC):
    """Abstract base class for domain-specific messaging channels."""
    
    def __init__(self, broker: MessageBroker, namespace: str):
        """
        Initialize channel with a message broker and namespace.
        
        Args:
            broker: Message broker instance
            namespace: Channel namespace (e.g., "training", "trading")
        """
        self.broker = broker
        self.namespace = namespace
    
    def _topic(self, subtopic: str) -> str:
        """
        Create namespaced topic.
        
        Args:
            subtopic: Subtopic name
            
        Returns:
            Namespaced topic string
        """
        return f"{self.namespace}.{subtopic}"
    
    def _key(self, resource: str, resource_id: str, field: Optional[str] = None) -> str:
        """
        Create namespaced storage key.
        
        Args:
            resource: Resource type (e.g., "experiment")
            resource_id: Resource identifier
            field: Optional field name
            
        Returns:
            Namespaced storage key
        """
        key = f"{self.namespace}:{resource}:{resource_id}"
        if field:
            key = f"{key}:{field}"
        return key
    
    async def publish(self, subtopic: str, data: Dict[str, Any]) -> None:
        """
        Publish message to namespaced topic.
        
        Args:
            subtopic: Subtopic to publish to
            data: Message data
        """
        topic = self._topic(subtopic)
        await self.broker.publish(topic, data)
    
    async def subscribe(self, subtopic_pattern: str) -> AsyncIterator[Message]:
        """
        Subscribe to namespaced topic pattern.
        
        Args:
            subtopic_pattern: Subtopic pattern (e.g., "experiment.*")
            
        Yields:
            Messages matching the pattern
        """
        topic_pattern = self._topic(subtopic_pattern)
        async for message in self.broker.subscribe(topic_pattern):
            yield message
    
    async def health_check(self) -> bool:
        """Check if the underlying broker is healthy."""
        return await self.broker.health_check()
    
    async def close(self) -> None:
        """Close the channel."""
        await self.broker.close() 