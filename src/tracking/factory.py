"""ML Tracking Factory System.

Simplified factory focused on Aim backend with clean generic architecture.
"""

from typing import Optional
import os

from src.tracking.protocols import MLTracker, ExperimentRepository, TrackingBackend
from src.tracking.backends.aim_backend import AimBackend
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global instances (singletons)
_tracker_instance: Optional[MLTracker] = None
_repository_instance: Optional[ExperimentRepository] = None
_current_backend: Optional[TrackingBackend] = None

def get_backend_config() -> dict[str, str]:
    """Get Aim backend configuration from environment."""
    return {
        "experiment_name": os.getenv("ML_EXPERIMENT_NAME", "AgenticTrading"),
        "storage_path": os.getenv("ML_STORAGE_PATH", ".aim"),
        "server_url": os.getenv("ML_SERVER_URL", ""),
    }

async def _get_backend() -> TrackingBackend:
    """Get Aim backend instance."""
    global _current_backend
    
    if _current_backend is None:
        _current_backend = AimBackend()
        await _current_backend.initialize()
        logger.info("Initialized ML tracking backend")
    
    return _current_backend

async def get_ml_tracker() -> MLTracker:
    """Get the Aim tracker instance (singleton)."""
    global _tracker_instance
    
    if _tracker_instance is None:
        backend = await _get_backend()
        _tracker_instance = backend.create_tracker()
    
    return _tracker_instance

async def get_experiment_repository() -> ExperimentRepository:
    """Get the Aim repository instance (singleton)."""
    global _repository_instance
    
    if _repository_instance is None:
        backend = await _get_backend()
        _repository_instance = backend.create_repository()
    
    return _repository_instance

async def create_tracker() -> MLTracker:
    """Create a new Aim tracker instance (not singleton)."""
    backend = AimBackend()
    await backend.initialize()
    return backend.create_tracker()

async def create_repository() -> ExperimentRepository:
    """Create a new Aim repository instance (not singleton)."""
    backend = AimBackend()
    await backend.initialize()
    return backend.create_repository()

async def reset_singletons() -> None:
    """Reset singleton instances (useful for testing)."""
    global _tracker_instance, _repository_instance, _current_backend
    
    _tracker_instance = None
    _repository_instance = None
    _current_backend = None
    
    logger.info("Reset ML tracking singletons")

async def health_check() -> dict[str, any]:
    """Perform health check on Aim backend."""
    try:
        backend = await _get_backend()
        health = await backend.health_check()
        return health.to_dict()
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "backend_name": "aim",
            "is_healthy": False,
            "error_message": str(e),
            "tracker_healthy": False,
            "repository_healthy": False,
            "storage_healthy": False
        }

# Convenience functions for configuration
async def configure_aim_backend(
    experiment_name: str = "AgenticTrading",
    storage_path: str = ".aim"
) -> None:
    """Configure Aim backend settings."""
    os.environ["ML_EXPERIMENT_NAME"] = experiment_name
    os.environ["ML_STORAGE_PATH"] = storage_path
    
    # Reset singletons to pick up new config
    await reset_singletons()
    
    logger.info(f"Configured Aim backend: {experiment_name} -> {storage_path}") 