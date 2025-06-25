"""Configuration for messaging system."""

import os
from dataclasses import dataclass, field
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class RedisConfig:
    """Redis broker configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0

@dataclass
class MemoryConfig:
    """In-memory broker configuration."""
    max_queue_size: int = 1000
    max_subscribers: int = 100

@dataclass
class MessagingConfig:
    """Main messaging configuration."""
    broker_type: str = "redis"  # "redis" or "memory"
    redis: RedisConfig = field(default_factory=RedisConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

def get_messaging_config() -> MessagingConfig:
    """
    Get messaging configuration from environment variables with fallback to defaults.
    
    Environment Variables:
        MESSAGE_BROKER_TYPE: "redis" or "memory" (default: "memory")
        REDIS_HOST: Redis host (default: "localhost")
        REDIS_PORT: Redis port (default: 6379)
        REDIS_DB: Redis database (default: 0)
        REDIS_PASSWORD: Redis password (default: None)
    
    Returns:
        MessagingConfig instance
    """
    broker_type = os.getenv("MESSAGE_BROKER_TYPE", "redis")  # Default to Redis broker
    
    redis_config = RedisConfig(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=int(os.getenv("REDIS_DB", "0")),
        password=os.getenv("REDIS_PASSWORD"),
        socket_timeout=float(os.getenv("REDIS_SOCKET_TIMEOUT", "5.0")),
        socket_connect_timeout=float(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5.0"))
    )
    
    memory_config = MemoryConfig(
        max_queue_size=int(os.getenv("MEMORY_MAX_QUEUE_SIZE", "1000")),
        max_subscribers=int(os.getenv("MEMORY_MAX_SUBSCRIBERS", "100"))
    )
    
    logger.info(f"Using {broker_type} message broker")
    
    return MessagingConfig(
        broker_type=broker_type,
        redis=redis_config,
        memory=memory_config
    ) 