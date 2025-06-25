"""In-memory message broker for development and testing."""

import asyncio
import fnmatch
import time
from collections import defaultdict, deque
from typing import Any, AsyncIterator, Dict, List, Optional

from src.messaging.base import MessageBroker, Message
from src.messaging.config import MemoryConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MemoryBroker(MessageBroker):
    """In-memory message broker using asyncio queues with data storage."""
    
    def __init__(self, config: MemoryConfig):
        """
        Initialize in-memory broker.
        
        Args:
            config: Memory broker configuration
        """
        self.config = config
        self._subscribers: Dict[str, List[asyncio.Queue]] = defaultdict(list)
        self._message_history: deque = deque(maxlen=config.max_queue_size)
        self._experiments: Dict[str, Dict[str, Any]] = {}  # In-memory experiment storage
        self._closed = False
        
        logger.info(f"Initialized MemoryBroker (max_queue_size: {config.max_queue_size})")
    
    async def publish(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Publish a message to a topic.
        
        Args:
            topic: Topic to publish to
            data: Message data dictionary
        """
        if self._closed:
            logger.warning("Cannot publish to closed broker")
            return
        
        message = Message(
            topic=topic,
            data=data,
            timestamp=time.time()
        )
        
        # Store in history
        self._message_history.append(message)
        
        # Send to matching subscribers
        delivered = 0
        for pattern, queues in self._subscribers.items():
            if fnmatch.fnmatch(topic, pattern):
                for queue in queues[:]:  # Copy list to avoid modification during iteration
                    try:
                        queue.put_nowait(message)
                        delivered += 1
                    except asyncio.QueueFull:
                        logger.warning(f"Queue full for pattern {pattern}, dropping message")
                    except Exception as e:
                        logger.error(f"Error delivering message to subscriber: {e}")
                        # Remove broken queue
                        if queue in queues:
                            queues.remove(queue)
        
        logger.debug(f"Published message to {topic}, delivered to {delivered} subscribers")
    
    async def subscribe(self, pattern: str) -> AsyncIterator[Message]:
        """
        Subscribe to messages matching a pattern.
        
        Args:
            pattern: Topic pattern (supports wildcards like "training.*")
            
        Yields:
            Message objects as they arrive
        """
        if self._closed:
            logger.warning("Cannot subscribe to closed broker")
            return
        
        if len(self._subscribers[pattern]) >= self.config.max_subscribers:
            logger.error(f"Max subscribers ({self.config.max_subscribers}) reached for pattern {pattern}")
            return
        
        queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        self._subscribers[pattern].append(queue)
        
        logger.info(f"New subscriber for pattern: {pattern}")
        
        try:
            while not self._closed:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield message
                except asyncio.TimeoutError:
                    continue  # Check if closed and continue waiting
                except Exception as e:
                    logger.error(f"Error in subscription loop: {e}")
                    break
        finally:
            # Clean up subscription
            if queue in self._subscribers[pattern]:
                self._subscribers[pattern].remove(queue)
            logger.info(f"Unsubscribed from pattern: {pattern}")
    
    # === DATA STORAGE METHODS ===
    # In-memory implementation matching Redis broker interface
    
    async def store_experiment(self, experiment_id: str, experiment_data: Dict[str, Any]) -> None:
        """
        Store experiment data in memory.
        
        Args:
            experiment_id: Unique experiment identifier
            experiment_data: Complete experiment data
        """
        if self._closed:
            logger.warning("Cannot store experiment in closed broker")
            return
        
        # Deep copy to avoid reference issues
        import copy
        self._experiments[experiment_id] = copy.deepcopy(experiment_data)
        logger.debug(f"Stored experiment {experiment_id}")
    
    async def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment data from memory.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Experiment data or None if not found
        """
        if self._closed:
            return None
        
        experiment_data = self._experiments.get(experiment_id)
        if experiment_data:
            # Return a copy to avoid reference issues
            import copy
            return copy.deepcopy(experiment_data)
        return None
    
    async def update_experiment(self, experiment_id: str, updates: Dict[str, Any]) -> None:
        """
        Update specific fields of an experiment.
        
        Args:
            experiment_id: Experiment identifier
            updates: Fields to update
        """
        if self._closed:
            logger.warning("Cannot update experiment in closed broker")
            return
        
        if experiment_id not in self._experiments:
            logger.warning(f"Experiment {experiment_id} not found for update")
            return
        
        # Update fields
        self._experiments[experiment_id].update(updates)
        logger.debug(f"Updated experiment {experiment_id}: {list(updates.keys())}")
    
    async def list_experiments(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all experiments, optionally filtered by status.
        
        Args:
            status_filter: Optional status to filter by
            
        Returns:
            List of experiment data
        """
        if self._closed:
            return []
        
        experiments = []
        for experiment_data in self._experiments.values():
            if status_filter is None or experiment_data.get("status") == status_filter:
                # Return a copy
                import copy
                experiments.append(copy.deepcopy(experiment_data))
        
        # Sort by start_time (most recent first)
        experiments.sort(
            key=lambda x: x.get("start_time", 0) or 0,
            reverse=True
        )
        
        return experiments
    
    async def remove_experiment(self, experiment_id: str) -> None:
        """
        Remove experiment from memory.
        
        Args:
            experiment_id: Experiment identifier
        """
        if self._closed:
            return
        
        if experiment_id in self._experiments:
            del self._experiments[experiment_id]
            logger.debug(f"Removed experiment {experiment_id}")
    
    async def close(self) -> None:
        """Close the message broker."""
        self._closed = True
        
        # Clear all queues to unblock subscribers
        for pattern_queues in self._subscribers.values():
            for queue in pattern_queues:
                try:
                    # Put None to signal end
                    queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass
        
        self._subscribers.clear()
        self._experiments.clear()
        logger.info("MemoryBroker closed")
    
    async def health_check(self) -> bool:
        """
        Check if the broker is healthy.
        
        Returns:
            True if broker is responsive
        """
        return not self._closed
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get broker statistics.
        
        Returns:
            Dictionary with broker stats
        """
        total_subscribers = sum(len(queues) for queues in self._subscribers.values())
        
        return {
            "broker_type": "memory",
            "total_subscribers": total_subscribers,
            "active_patterns": len(self._subscribers),
            "message_history_size": len(self._message_history),
            "max_queue_size": self.config.max_queue_size,
            "max_subscribers": self.config.max_subscribers,
            "active_experiments": len(self._experiments),
            "closed": self._closed
        } 