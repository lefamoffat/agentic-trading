import asyncio
from typing import AsyncGenerator, Optional

from apps.api.core.redis_stream import StreamManager
from apps.api.core.experiments_service import UnifiedExperimentService


# ---------------------------------------------------------------------------
# Lazy singletons – async-friendly (cannot rely on lru_cache with coroutines)
# ---------------------------------------------------------------------------

_stream_manager: Optional[StreamManager] = None
_experiments_service: Optional[UnifiedExperimentService] = None


async def get_stream_manager() -> StreamManager:  # noqa: WPS231
    """Return a singleton StreamManager instance (initialized on first call)."""
    global _stream_manager

    if _stream_manager is None:
        _stream_manager = StreamManager()
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

async def stream_manager_dep() -> AsyncGenerator[StreamManager, None]:
    yield await get_stream_manager() 