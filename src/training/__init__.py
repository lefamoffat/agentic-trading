"""Training module with real-time messaging support."""

def __getattr__(name):
    """Lazy import to avoid configuration loading during test imports."""
    if name == "TrainingService":
        from src.training.service import TrainingService
        return TrainingService
    elif name == "get_training_service":
        from src.training.service import get_training_service
        return get_training_service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["TrainingService", "get_training_service"] 