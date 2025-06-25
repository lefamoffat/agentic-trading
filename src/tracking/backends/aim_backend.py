"""Aim ML Tracking Backend.

Production implementation using Aim for ML experiment tracking.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import os

try:
    from aim import Run, Repo
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False
    Run = None
    Repo = None

from src.tracking.protocols import (
    MLTracker, ExperimentRepository, ExperimentRun, MetricSeries, TrackingBackend
)
from src.tracking.models import (
    TrainingMetrics, ExperimentSummary, ModelArtifact, ExperimentConfig,
    SystemHealth, ExperimentStatus
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

class AimExperimentRun:
    """Aim implementation of ExperimentRun."""
    
    def __init__(self, aim_run: 'Run'):
        self._aim_run = aim_run
        self._run_hash = aim_run.hash
        self._experiment_id = aim_run.get("experiment_id", "unknown")
        
    @property
    def id(self) -> str:
        return self._run_hash
    
    @property
    def experiment_id(self) -> str:
        return self._experiment_id
    
    @property
    def status(self) -> str:
        return "running" if self._aim_run.active else "completed"
    
    @property
    def start_time(self) -> Optional[datetime]:
        if hasattr(self._aim_run, 'creation_time'):
            return datetime.fromtimestamp(self._aim_run.creation_time)
        return None
    
    @property
    def end_time(self) -> Optional[datetime]:
        if hasattr(self._aim_run, 'end_time') and self._aim_run.end_time:
            return datetime.fromtimestamp(self._aim_run.end_time)
        return None

class AimMLTracker:
    """Aim implementation of MLTracker."""
    
    def __init__(self, aim_repo: 'Repo'):
        self.repo = aim_repo
        self.logger = get_logger(self.__class__.__name__)
        self._active_runs: Dict[str, Run] = {}
    
    async def start_run(
        self,
        experiment_id: str,
        config: ExperimentConfig,
        run_name: Optional[str] = None
    ) -> ExperimentRun:
        """Start a new training run."""
        aim_run = Run(repo=self.repo, experiment=experiment_id)
        
        # Set metadata
        aim_run["experiment_id"] = experiment_id
        aim_run["run_name"] = run_name or f"Run_{aim_run.hash[:8]}"
        aim_run["agent_type"] = config.agent_type
        
        if config.symbol:
            aim_run["symbol"] = config.symbol
        if config.timesteps:
            aim_run["timesteps"] = config.timesteps
        if config.learning_rate:
            aim_run["learning_rate"] = config.learning_rate
        
        self._active_runs[aim_run.hash] = aim_run
        wrapped_run = AimExperimentRun(aim_run)
        
        self.logger.info(f"Started Aim run: {wrapped_run.id}")
        return wrapped_run
    
    async def log_metrics(
        self,
        run_id: str,
        metrics: Dict[str, float],
        step: int,
        context: Optional[Dict[str, str]] = None
    ) -> None:
        """Log training metrics for a step."""
        if run_id not in self._active_runs:
            self.logger.warning(f"Run {run_id} not found")
            return
        
        aim_run = self._active_runs[run_id]
        
        for name, value in metrics.items():
            metric_context = {"subset": "train"}
            if context:
                metric_context.update(context)
            
            aim_run.track(
                value=float(value),
                name=name,
                step=step,
                context=metric_context
            )
    
    async def log_training_metrics(
        self,
        run_id: str,
        training_metrics: TrainingMetrics,
        step: int
    ) -> None:
        """Log structured trading metrics."""
        metrics_dict = training_metrics.to_dict()
        await self.log_metrics(run_id, metrics_dict, step, {"type": "training"})
    
    async def log_hyperparameters(
        self,
        run_id: str,
        hyperparams: Dict[str, Any]
    ) -> None:
        """Log hyperparameters for the run."""
        if run_id not in self._active_runs:
            return
        
        aim_run = self._active_runs[run_id]
        for name, value in hyperparams.items():
            aim_run[f"hparams.{name}"] = value
    
    async def log_model_artifact(
        self,
        run_id: str,
        artifact: ModelArtifact
    ) -> None:
        """Log model artifacts."""
        if run_id not in self._active_runs:
            return
        
        aim_run = self._active_runs[run_id]
        aim_run[f"artifacts.{artifact.name}"] = {
            "type": artifact.artifact_type.value,
            "metadata": artifact.metadata
        }
    
    async def finalize_run(
        self,
        run_id: str,
        final_metrics: Optional[Dict[str, Any]] = None,
        status: str = "completed"
    ) -> None:
        """Finalize and close the run."""
        if run_id not in self._active_runs:
            return
        
        aim_run = self._active_runs[run_id]
        
        if final_metrics:
            for name, value in final_metrics.items():
                aim_run[f"final.{name}"] = value
        
        aim_run["status"] = status
        aim_run.close()
        del self._active_runs[run_id]
        
        self.logger.info(f"Finalized Aim run {run_id}")
    
    async def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Get a specific run by ID."""
        if run_id in self._active_runs:
            return AimExperimentRun(self._active_runs[run_id])
        return None

class AimExperimentRepository:
    """Aim implementation of ExperimentRepository."""
    
    def __init__(self, aim_repo: 'Repo'):
        self.repo = aim_repo
        self.logger = get_logger(self.__class__.__name__)
    
    async def get_recent_experiments(
        self,
        limit: int = 10,
        experiment_name: Optional[str] = None
    ) -> List[ExperimentSummary]:
        """Get recent experiments with summary data."""
        try:
            query = "run.archived == False"
            if experiment_name:
                query += f" and run.experiment == '{experiment_name}'"
            
            runs = list(self.repo.query_runs(query).limit(limit))
            
            experiments = []
            for run in runs:
                summary = self._create_experiment_summary(run)
                if summary:
                    experiments.append(summary)
            
            return experiments
        except Exception as e:
            self.logger.error(f"Failed to get recent experiments: {e}")
            return []
    
    async def get_experiment_details(
        self,
        experiment_id: str
    ) -> Optional[ExperimentSummary]:
        """Get detailed information about a specific experiment."""
        try:
            runs = list(self.repo.query_runs(f"run.experiment == '{experiment_id}'").limit(1))
            
            if runs:
                return self._create_experiment_summary(runs[0], detailed=True)
        except Exception as e:
            self.logger.error(f"Failed to get experiment details: {e}")
        
        return None
    
    async def get_metric_history(
        self,
        run_id: str,
        metric_name: str
    ) -> Optional[MetricSeries]:
        """Get time-series data for a specific metric."""
        # Simplified for now - would need full Aim query implementation
        return None
    
    async def search_experiments(
        self,
        query: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        limit: int = 50
    ) -> List[ExperimentSummary]:
        """Search experiments with filters."""
        return await self.get_recent_experiments(limit)
    
    async def get_system_health(self) -> SystemHealth:
        """Get system health and connection status."""
        try:
            all_runs = list(self.repo.query_runs(""))
            total_runs = len(all_runs)
            active_runs = [r for r in all_runs if r.active]
            
            return SystemHealth(
                backend_name="aim",
                is_healthy=True,
                last_check=datetime.now(),
                tracker_healthy=True,
                repository_healthy=True,
                storage_healthy=True,
                total_experiments=len(set(r.experiment for r in all_runs)),
                active_experiments=len(set(r.experiment for r in active_runs)),
                total_runs=total_runs
            )
        except Exception as e:
            return SystemHealth(
                backend_name="aim",
                is_healthy=False,
                error_message=str(e),
                tracker_healthy=False,
                repository_healthy=False,
                storage_healthy=False
            )
    
    async def get_experiment_runs(
        self,
        experiment_id: str,
        limit: int = 50
    ) -> List[ExperimentSummary]:
        """Get all runs for a specific experiment."""
        try:
            runs = list(self.repo.query_runs(f"run.experiment == '{experiment_id}'").limit(limit))
            
            experiments = []
            for run in runs:
                summary = self._create_experiment_summary(run)
                if summary:
                    experiments.append(summary)
            
            return experiments
        except Exception as e:
            self.logger.error(f"Failed to get experiment runs: {e}")
            return []
    
    def _create_experiment_summary(
        self,
        aim_run: 'Run',
        detailed: bool = False
    ) -> Optional[ExperimentSummary]:
        """Create ExperimentSummary from Aim run."""
        try:
            run_id = aim_run.hash
            experiment_id = aim_run.get("experiment_id", aim_run.experiment)
            
            status = ExperimentStatus.RUNNING if aim_run.active else ExperimentStatus.COMPLETED
            
            start_time = None
            if hasattr(aim_run, 'creation_time'):
                start_time = datetime.fromtimestamp(aim_run.creation_time)
            
            return ExperimentSummary(
                experiment_id=experiment_id,
                run_id=run_id,
                name=aim_run.get("run_name", f"Run_{run_id[:8]}"),
                status=status,
                start_time=start_time,
                agent_type=aim_run.get("agent_type", "unknown"),
                symbol=aim_run.get("symbol"),
                timesteps=aim_run.get("timesteps"),
                backend_name="aim"
            )
        except Exception as e:
            self.logger.error(f"Failed to create experiment summary: {e}")
            return None

class AimBackend(TrackingBackend):
    """Aim tracking backend for production use."""
    
    def __init__(self):
        if not AIM_AVAILABLE:
            raise ImportError("Aim package not available. Install with: uv add aim")
        
        self.repo_path = os.getenv("ML_STORAGE_PATH", ".aim")
        self.repo: Optional[Repo] = None
        self.logger = get_logger(self.__class__.__name__)
    
    def create_tracker(self) -> MLTracker:
        """Create a tracker instance."""
        if not self.repo:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        return AimMLTracker(self.repo)
    
    def create_repository(self) -> ExperimentRepository:
        """Create a repository instance."""
        if not self.repo:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        return AimExperimentRepository(self.repo)
    
    async def initialize(self) -> None:
        """Initialize the backend."""
        from pathlib import Path
        
        try:
            # Try to create/open the repository
            # If directory doesn't exist, Aim will create it with proper schema
            self.repo = Repo(path=self.repo_path, init=True)
        except Exception as e:
            # If that fails, ensure directory exists and try manual init
            aim_dir = Path(self.repo_path)
            aim_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Force initialize an empty repository
                self.repo = Repo(path=str(aim_dir), init=True)
            except Exception as init_error:
                self.logger.error(f"Failed to initialize Aim repository: {init_error}")
                raise RuntimeError(f"Could not initialize Aim backend at {self.repo_path}: {init_error}")
        
        # Verify the repository is working by trying a basic operation
        try:
            # This will trigger schema creation if it doesn't exist
            list(self.repo.query_runs("").limit(1))
        except Exception as e:
            self.logger.warning(f"Repository verification failed, but continuing: {e}")
        
        self.logger.info(f"Initialized Aim backend with repository at: {self.repo_path}")
    
    async def health_check(self) -> SystemHealth:
        """Check backend health."""
        if not self.repo:
            return SystemHealth(
                backend_name="aim",
                is_healthy=False,
                error_message="Backend not initialized"
            )
        
        repository = self.create_repository()
        return await repository.get_system_health()
    
    @property
    def name(self) -> str:
        """Backend name."""
        return "aim"
