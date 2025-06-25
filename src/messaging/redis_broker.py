"""Redis-based message broker for production use."""

import asyncio
import json
import time
import fnmatch
from typing import Any, AsyncIterator, Dict, Optional, List
import traceback

import redis.asyncio as redis
from redis.exceptions import ConnectionError, TimeoutError

from src.messaging.base import MessageBroker, Message
from src.messaging.config import RedisConfig
from src.utils.logger import get_logger
from src.messaging.schema import ExperimentState

logger = get_logger(__name__)

class RedisBroker(MessageBroker):
    """Redis-based message broker using pub/sub with data storage capabilities."""
    
    def __init__(self, config: RedisConfig):
        """
        Initialize Redis broker.
        
        Args:
            config: Redis broker configuration
        """
        self.config = config
        self._redis: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self._subscriptions: Dict[str, bool] = {}
        self._running = False
        
        logger.info(f"Initialized RedisBroker ({config.host}:{config.port})")
        
    async def connect(self) -> None:
        """Connect to Redis server."""
        if self._redis is not None:
            return
            
        try:
            logger.info(f"[REDIS] Attempting to connect to Redis at {self.config.host}:{self.config.port}")
            self._redis = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                decode_responses=True,
                protocol=3  # Use RESP3 for better performance
            )
            
            # Test connection
            logger.info("[REDIS] Testing connection with PING")
            try:
                await self._redis.ping()
            except (ConnectionError, TimeoutError) as e:
                logger.error(f"[REDIS] Failed to ping Redis: {e}")
                logger.error(f"[REDIS] Exception type: {type(e).__name__}")
                logger.error(f"[REDIS] Traceback: {traceback.format_exc()}")
                self._redis = None
                raise
            
            # Create PubSub instance
            self._pubsub = self._redis.pubsub()
            
            logger.info(f"[REDIS] Successfully connected to Redis at {self.config.host}:{self.config.port}")
            
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"[REDIS] Failed to connect to Redis: {e}")
            logger.error(f"[REDIS] Exception type: {type(e).__name__}")
            logger.error(f"[REDIS] Traceback: {traceback.format_exc()}")
            self._redis = None
            raise
        except Exception as e:
            logger.error(f"[REDIS] Unexpected error connecting to Redis: {e}")
            logger.error(f"[REDIS] Exception type: {type(e).__name__}")
            logger.error(f"[REDIS] Traceback: {traceback.format_exc()}")
            self._redis = None
            raise
    
    async def close(self) -> None:
        """Close the message broker connection."""
        await self.disconnect()
    
    async def disconnect(self) -> None:
        """Disconnect from Redis server."""
        self._running = False
        
        if self._pubsub:
            try:
                await self._pubsub.aclose()
            except Exception as e:
                logger.warning(f"Error closing pubsub: {e}")
            finally:
                self._pubsub = None
        
        if self._redis:
            try:
                await self._redis.aclose()
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")
            finally:
                self._redis = None
                
        logger.info("Disconnected from Redis")
    
    async def publish(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Publish a message to a topic.
        
        Args:
            topic: Topic to publish to (e.g., "training.status")
            data: Message data dictionary
        """
        if not self._redis:
            await self.connect()
        
        message = {
            "topic": topic,
            "data": data,
            "timestamp": time.time()
        }
        
        try:
            message_json = json.dumps(message)
            subscribers = await self._redis.publish(topic, message_json)
            logger.debug(f"Published to {topic}: {data} ({subscribers} subscribers)")
            
        except Exception as e:
            logger.error(f"Failed to publish to {topic}: {e}")
            raise
    
    async def subscribe(self, pattern: str) -> AsyncIterator[Message]:
        """
        Subscribe to topics matching a pattern.
        
        Args:
            pattern: Topic pattern (e.g., "training.*")
            
        Yields:
            Message: Received messages
        """
        if not self._redis:
            await self.connect()
        
        if not self._pubsub:
            self._pubsub = self._redis.pubsub()
        
        try:
            # Use psubscribe for pattern matching
            await self._pubsub.psubscribe(pattern)
            self._subscriptions[pattern] = True
            self._running = True
            
            logger.info(f"Subscribed to pattern: {pattern}")
            
            # Listen for messages
            async for raw_message in self._pubsub.listen():
                if not self._running:
                    break
                    
                # Skip subscription confirmation messages
                if raw_message["type"] in ("psubscribe", "punsubscribe"):
                    continue
                
                if raw_message["type"] == "pmessage":
                    try:
                        # Parse JSON message
                        message_data = json.loads(raw_message["data"])
                        
                        message = Message(
                            topic=message_data["topic"],
                            data=message_data["data"],
                            timestamp=message_data.get("timestamp")
                        )
                        
                        yield message
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Failed to parse message: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error in subscription to {pattern}: {e}")
            raise
        finally:
            if pattern in self._subscriptions:
                del self._subscriptions[pattern]
    
    async def unsubscribe(self, pattern: str) -> None:
        """
        Unsubscribe from a topic pattern.
        
        Args:
            pattern: Topic pattern to unsubscribe from
        """
        if self._pubsub and pattern in self._subscriptions:
            try:
                await self._pubsub.punsubscribe(pattern)
                del self._subscriptions[pattern]
                logger.info(f"Unsubscribed from pattern: {pattern}")
            except Exception as e:
                logger.error(f"Failed to unsubscribe from {pattern}: {e}")
                raise
    
    # === DATA STORAGE METHODS ===
    # Make Redis the single source of truth for experiment data
    
    async def store_experiment(self, experiment_id: str, experiment_data: Dict[str, Any] | ExperimentState) -> None:
        """
        Store experiment data in Redis.
        
        Args:
            experiment_id: Unique experiment identifier
            experiment_data: Complete experiment data
        """
        if not self._redis:
            await self.connect()
        
        try:
            key = f"experiment:{experiment_id}"

            if isinstance(experiment_data, ExperimentState):
                flattened_data = experiment_data.to_flat_dict()
            else:
                experiment_data["id"] = experiment_id
                flattened_data = {k: json.dumps(v) if isinstance(v, dict) else str(v) for k, v in experiment_data.items()}

            await self._redis.hset(key, mapping=flattened_data)
            
            # Add to active experiments set
            await self._redis.sadd("active_experiments", experiment_id)

            # Determine status string
            status_value = (
                experiment_data.status if isinstance(experiment_data, ExperimentState) else experiment_data.get("status")
            )

            # Set expiration (24 hours for terminal statuses)
            if status_value in {"completed", "failed", "cancelled"}:
                await self._redis.expire(key, 86400)
            
            logger.debug(f"Stored experiment {experiment_id}")
            
        except Exception as e:
            logger.error(f"Failed to store experiment {experiment_id}: {e}")
            raise
    
    async def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment data from Redis.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Experiment data or None if not found
        """
        if not self._redis:
            await self.connect()
        
        try:
            key = f"experiment:{experiment_id}"
            logger.info(f"[REDIS] Getting experiment data for {key}")
            
            data = await self._redis.hgetall(key)
            logger.info(f"[REDIS] Raw data for {experiment_id}: {data}")
            
            if not data:
                logger.warning(f"[REDIS] No data found for experiment {experiment_id}")
                return None
            
            try:
                state = ExperimentState.from_redis_hash(data)
            except Exception as parse_err:
                logger.error(f"[REDIS] Failed to parse experiment {experiment_id}: {parse_err}")
                return None
            logger.info(f"[REDIS] Processed data for {experiment_id}: {state.status}")
            return state.model_dump()
            
        except Exception as e:
            logger.error(f"[REDIS] Failed to get experiment {experiment_id}: {e}")
            return None
    
    async def update_experiment(self, experiment_id: str, updates: Dict[str, Any]) -> None:
        """
        Update specific fields of an experiment.
        
        Args:
            experiment_id: Experiment identifier
            updates: Fields to update
        """
        if not self._redis:
            await self.connect()
        
        logger.debug(f"[REDIS] update_experiment called with updates={updates}")
        
        try:
            key = f"experiment:{experiment_id}"
            
            # Check if experiment exists
            exists = await self._redis.exists(key)
            logger.info(f"[REDIS] !!!! Checking existence of {key}: {exists} !!!!")
            if not exists:
                logger.error(f"[REDIS] !!!! Experiment {experiment_id} not found for update !!!!")
                return
            
            # Flatten updates
            flattened_updates = {}
            for field, value in updates.items():
                if isinstance(value, dict):
                    flattened_updates[field] = json.dumps(value)
                elif field == "status" and hasattr(value, "value"):
                    # Handle TrainingStatus enum
                    flattened_updates[field] = value.value
                    logger.info(f"[REDIS] !!!! Converting status enum {value} to value {value.value} !!!!")
                else:
                    flattened_updates[field] = str(value)
            
            logger.info(f"[REDIS] !!!! Updating experiment {experiment_id} with fields: {list(updates.keys())} !!!!")
            logger.info(f"[REDIS] !!!! Update values: {flattened_updates} !!!!")
            
            # Perform update
            try:
                await self._redis.hset(key, mapping=flattened_updates)
                logger.info(f"[REDIS] !!!! Successfully updated experiment {experiment_id} !!!!")
                
                # Verify update
                updated_data = await self._redis.hgetall(key)
                logger.info(f"[REDIS] !!!! Verified update for {experiment_id}: {updated_data} !!!!")
            except Exception as redis_error:
                logger.error(f"[REDIS] !!!! Redis operation failed: {redis_error} !!!!")
                logger.error(f"[REDIS] !!!! Redis error type: {type(redis_error).__name__} !!!!")
                logger.error(f"[REDIS] !!!! Redis error traceback:\n{traceback.format_exc()} !!!!")
                raise
            
        except Exception as e:
            logger.error(f"[REDIS] !!!! Failed to update experiment {experiment_id}: {e} !!!!")
            logger.error(f"[REDIS] !!!! Error type: {type(e).__name__} !!!!")
            logger.error(f"[REDIS] !!!! Error traceback:\n{traceback.format_exc()} !!!!")
            raise
    
    async def list_experiments(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all experiments, optionally filtered by status.
        
        Args:
            status_filter: Optional status to filter by
            
        Returns:
            List of experiment data
        """
        if not self._redis:
            await self.connect()
        
        try:
            # Get all active experiment IDs
            experiment_ids = await self._redis.smembers("active_experiments")
            
            experiments = []
            for experiment_id in experiment_ids:
                experiment_data = await self.get_experiment(experiment_id)
                if experiment_data:
                    if status_filter is None or experiment_data.get("status") == status_filter:
                        experiments.append(experiment_data)
            
            # Sort by start_time (most recent first)
            experiments.sort(
                key=lambda x: x.get("start_time", 0) or 0,
                reverse=True
            )
            
            return experiments
            
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return []
    
    async def remove_experiment(self, experiment_id: str) -> None:
        """
        Remove experiment from active set (but keep data for history).
        
        Args:
            experiment_id: Experiment identifier
        """
        if not self._redis:
            await self.connect()
        
        try:
            # Remove from active set
            await self._redis.srem("active_experiments", experiment_id)
            
            # Set expiration on the data (keep for 24 hours)
            key = f"experiment:{experiment_id}"
            await self._redis.expire(key, 86400)
            
            logger.debug(f"Removed experiment {experiment_id} from active set")
            
        except Exception as e:
            logger.error(f"Failed to remove experiment {experiment_id}: {e}")
            raise
    
    async def health_check(self) -> bool:
        """
        Check if Redis connection is healthy.
        
        Returns:
            bool: True if Redis is connected and responding
        """
        try:
            if not self._redis:
                await self.connect()
                
            await self._redis.ping()
            return True
            
        except Exception as e:
            logger.error(f"[REDIS] Health check failed: {e}")
            logger.error(f"[REDIS] Exception type: {type(e).__name__}")
            logger.error(f"[REDIS] Traceback: {traceback.format_exc()}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get broker statistics.
        
        Returns:
            Dict with broker statistics
        """
        if not self._redis:
            return {"connected": False}
        
        try:
            info = await self._redis.info()
            
            # Get experiment stats
            active_count = await self._redis.scard("active_experiments")
            
            return {
                "connected": True,
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory": info.get("used_memory"),
                "used_memory_human": info.get("used_memory_human"),
                "subscriptions": list(self._subscriptions.keys()),
                "active_experiments": active_count
            }
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {"connected": False, "error": str(e)} 