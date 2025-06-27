import asyncio
from typing import AsyncGenerator, Optional

from src.messaging.factory import get_message_broker
from src.messaging.relay import BrokerRelay
from apps.api.core.experiments_service import UnifiedExperimentService


# ---------------------------------------------------------------------------
# Lazy singletons – async-friendly (cannot rely on lru_cache with coroutines)
# ---------------------------------------------------------------------------

_stream_manager: Optional[BrokerRelay] = None
_experiments_service: Optional[UnifiedExperimentService] = None


async def get_stream_manager() -> BrokerRelay:  # noqa: WPS231
    """Return a singleton StreamManager instance (initialized on first call)."""
    global _stream_manager

    if _stream_manager is None:
        broker = get_message_broker()
        _stream_manager = BrokerRelay(broker)
        await _stream_manager.start()

    return _stream_manager


async def get_experiments_service() -> UnifiedExperimentService:  # noqa: WPS231
    """Return a singleton UnifiedExperimentService (initialized on first call)."""
    global _experiments_service

    if _experiments_service is None:
        _experiments_service = UnifiedExperimentService()
        await _experiments_service.initialize()

    return _experiments_service

# FastAPI dependency-friendly wrappers
async def experiments_service_dep() -> AsyncGenerator[UnifiedExperimentService, None]:
    yield await get_experiments_service()

async def stream_manager_dep() -> AsyncGenerator[BrokerRelay, None]:
    yield await get_stream_manager() 