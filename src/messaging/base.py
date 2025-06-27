"""Abstract base class for message brokers."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Optional, List
from dataclasses import dataclass
from src.types.experiments import Experiment

@dataclass
class Message:
    """Message container for pub/sub events."""
    topic: str
    data: Dict[str, Any]
    timestamp: Optional[float] = None

class MessageBroker(ABC):
    """Abstract base class for message brokers with pub/sub and data storage."""
    
    # === PUB/SUB METHODS ===
    
    @abstractmethod
    async def publish(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Publish a message to a topic.
        
        Args:
            topic: Topic to publish to (e.g., "training.status")
            data: Message data dictionary
        """
        pass
    
    @abstractmethod
    async def subscribe(self, pattern: str) -> AsyncIterator[Message]:
        """
        Subscribe to messages matching a pattern.
        
        Args:
            pattern: Topic pattern (e.g., "training.*" for all training topics)
            
        Yields:
            Message objects as they arrive
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the message broker connection."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the message broker is healthy.
        
        Returns:
            True if broker is responsive, False otherwise
        """
        pass
    
    # === DATA STORAGE METHODS ===
    
    @abstractmethod
    async def store_experiment(self, exp: Experiment) -> None:
        """
        Store experiment data.
        
        Args:
            exp: Experiment object
        """
        pass
    
    @abstractmethod
    async def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """
        Get experiment data.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Experiment data or None if not found
        """
        pass
    
    @abstractmethod
    async def update_experiment(self, experiment_id: str, updates: Dict[str, Any]) -> None:
        """
        Update specific fields of an experiment.
        
        Args:
            experiment_id: Experiment identifier
            updates: Fields to update
        """
        pass
    
    @abstractmethod
    async def list_experiments(self, status_filter: Optional[Any] = None) -> List[Experiment]:
        """
        List all experiments, optionally filtered by status.
        
        Args:
            status_filter: Optional status to filter by
            
        Returns:
            List of experiment data
        """
        pass
    
    @abstractmethod
    async def remove_experiment(self, experiment_id: str) -> None:
        """
        Remove experiment from active storage.
        
        Args:
            experiment_id: Experiment identifier
        """
        pass 