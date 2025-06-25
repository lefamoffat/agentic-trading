"""ML Tracking Protocols.

These protocols define the interface that any ML tracking backend must implement.
This allows swapping between Aim, Weights & Biases, TensorBoard, etc. without
changing the calling code.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncIterator, Union, Protocol
from datetime import datetime

from src.tracking.models import (
    TrainingMetrics,
    ExperimentSummary,
    ModelArtifact,
    ExperimentConfig,
    SystemHealth
)

class ExperimentRun(Protocol):
    """Generic interface for a single experiment run."""
    
    @property
    def id(self) -> str:
        """Unique run identifier."""
        ...
    
    @property
    def experiment_id(self) -> str:
        """Parent experiment identifier."""
        ...
    
    @property
    def status(self) -> str:
        """Current run status (running, completed, failed, etc.)."""
        ...
    
    @property
    def start_time(self) -> Optional[datetime]:
        """When the run started."""
        ...
    
    @property
    def end_time(self) -> Optional[datetime]:
        """When the run ended (None if still running)."""
        ...

class MetricSeries(Protocol):
    """Generic interface for time-series metrics."""
    
    @property
    def name(self) -> str:
        """Metric name."""
        ...
    
    @property
    def values(self) -> List[float]:
        """List of metric values."""
        ...
    
    @property
    def steps(self) -> List[int]:
        """List of steps corresponding to values."""
        ...
    
    @property
    def context(self) -> Dict[str, str]:
        """Metric context (e.g., {"type": "reward", "subset": "train"})."""
        ...

class MLTracker(Protocol):
    """Generic interface for ML experiment tracking."""
    
    async def start_run(
        self,
        experiment_id: str,
        config: ExperimentConfig,
        run_name: Optional[str] = None
    ) -> ExperimentRun:
        """Start a new training run."""
        ...
    
    async def log_metrics(
        self,
        run_id: str,
        metrics: Dict[str, float],
        step: int,
        context: Optional[Dict[str, str]] = None
    ) -> None:
        """Log training metrics for a step."""
        ...
    
    async def log_training_metrics(
        self,
        run_id: str,
        training_metrics: TrainingMetrics,
        step: int
    ) -> None:
        """Log structured trading metrics."""
        ...
    
    async def log_hyperparameters(
        self,
        run_id: str,
        hyperparams: Dict[str, Any]
    ) -> None:
        """Log hyperparameters for the run."""
        ...
    
    async def log_model_artifact(
        self,
        run_id: str,
        artifact: ModelArtifact
    ) -> None:
        """Log model artifacts (saved models, plots, etc.)."""
        ...
    
    async def finalize_run(
        self,
        run_id: str,
        final_metrics: Optional[Dict[str, Any]] = None,
        status: str = "completed"
    ) -> None:
        """Finalize and close the run."""
        ...
    
    async def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Get a specific run by ID."""
        ...

class ExperimentRepository(Protocol):
    """Generic interface for querying experiment data."""
    
    async def get_recent_experiments(
        self,
        limit: int = 10,
        experiment_name: Optional[str] = None
    ) -> List[ExperimentSummary]:
        """Get recent experiments with summary data."""
        ...
    
    async def get_experiment_details(
        self,
        experiment_id: str
    ) -> Optional[ExperimentSummary]:
        """Get detailed information about a specific experiment."""
        ...
    
    async def get_metric_history(
        self,
        run_id: str,
        metric_name: str
    ) -> Optional[MetricSeries]:
        """Get time-series data for a specific metric."""
        ...
    
    async def search_experiments(
        self,
        query: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        limit: int = 50
    ) -> List[ExperimentSummary]:
        """Search experiments with filters."""
        ...
    
    async def get_system_health(self) -> SystemHealth:
        """Get system health and connection status."""
        ...
    
    async def get_experiment_runs(
        self,
        experiment_id: str,
        limit: int = 50
    ) -> List[ExperimentSummary]:
        """Get all runs for a specific experiment."""
        ...

class TrackingBackend(ABC):
    """Abstract base class for tracking backend implementations."""
    
    @abstractmethod
    def create_tracker(self) -> MLTracker:
        """Create a tracker instance."""
        pass
    
    @abstractmethod
    def create_repository(self) -> ExperimentRepository:
        """Create a repository instance."""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the backend."""
        pass
    
    @abstractmethod
    async def health_check(self) -> SystemHealth:
        """Check backend health."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name."""
        pass 