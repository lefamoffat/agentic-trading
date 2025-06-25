"""ML Tracking Module.

Provides clean interfaces for ML experiment tracking using Aim backend.
"""

from src.tracking.factory import (
    get_ml_tracker,
    get_experiment_repository,
    create_tracker,
    create_repository,
    configure_aim_backend,
    reset_singletons,
    health_check
)
from src.tracking.models import (
    TrainingMetrics,
    ExperimentSummary,
    ModelArtifact,
    ExperimentConfig,
    SystemHealth,
    ExperimentStatus,
    ArtifactType
)
from src.tracking.protocols import (
    MLTracker,
    ExperimentRepository,
    ExperimentRun,
    MetricSeries,
    TrackingBackend
)
from src.tracking.callbacks import (
    TrackingCallback,
    SimpleTrackingCallback
)

__all__ = [
    # Factory functions
    "get_ml_tracker",
    "get_experiment_repository", 
    "create_tracker",
    "create_repository",
    "configure_aim_backend",
    "reset_singletons",
    "health_check",
    
    # Data models
    "TrainingMetrics",
    "ExperimentSummary", 
    "ModelArtifact",
    "ExperimentConfig",
    "SystemHealth",
    "ExperimentStatus",
    "ArtifactType",
    
    # Protocols
    "MLTracker",
    "ExperimentRepository",
    "ExperimentRun", 
    "MetricSeries",
    "TrackingBackend",
    
    # Callbacks
    "TrackingCallback",
    "SimpleTrackingCallback",
] 