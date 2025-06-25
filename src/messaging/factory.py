"""Message broker factory."""

import asyncio

from src.messaging.base import MessageBroker
from src.messaging.config import get_messaging_config
from src.messaging.redis_broker import RedisBroker
from src.messaging.memory_broker import MemoryBroker
from src.utils.logger import get_logger

logger = get_logger(__name__)

_broker_instance: MessageBroker | None = None
_broker_loop: asyncio.AbstractEventLoop | None = None

def get_message_broker() -> MessageBroker:
    """
    Get configured message broker instance (singleton).
    
    Returns:
        Configured message broker (Redis or Memory based on config)
    """
    global _broker_instance, _broker_loop
    
    # Detect current loop (may be absent if called from sync context)
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    # If an instance exists but is bound to a different or closed loop, reset it
    if _broker_instance is not None:
        if _broker_loop is None or (_broker_loop is not current_loop and (_broker_loop.is_closed() if hasattr(_broker_loop, 'is_closed') else False)):
            logger.info("[BROKER] Existing broker bound to different/closed loop → recreating")
            _broker_instance = None

    if _broker_instance is None:
        logger.info("[BROKER] !!!! Getting messaging config !!!!")
        config = get_messaging_config()
        logger.info(f"[BROKER] !!!! Got config: broker_type={config.broker_type} !!!!")
        
        if config.broker_type == "redis":
            logger.info("[BROKER] !!!! Initializing Redis message broker !!!!")
            _broker_instance = RedisBroker(config.redis)
            logger.info("[BROKER] !!!! Redis message broker initialized !!!!")
        elif config.broker_type == "memory":
            logger.info("[BROKER] !!!! Initializing in-memory message broker !!!!")
            _broker_instance = MemoryBroker(config.memory)
            logger.info("[BROKER] !!!! In-memory message broker initialized !!!!")
        else:
            logger.error(f"[BROKER] !!!! Unknown broker type: {config.broker_type} !!!!")
            raise ValueError(f"Unknown broker type: {config.broker_type}")
    
    # Remember loop the broker was created under
    _broker_loop = current_loop

    return _broker_instance

def reset_broker():
    """Reset broker singleton (for testing)."""
    global _broker_instance, _broker_loop
    _broker_instance = None
    _broker_loop = None 