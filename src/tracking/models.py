"""ML Tracking Data Models.

These models represent experiment data in a backend-agnostic way.
They can be populated from Aim, Weights & Biases, or any other tracking system.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum

class ExperimentStatus(Enum):
    """Standard experiment status values."""
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ArtifactType(Enum):
    """Types of model artifacts."""
    MODEL = "model"
    PLOT = "plot"
    DATA = "data"
    LOG = "log"
    CONFIG = "config"

@dataclass
class TrainingMetrics:
    """Structured training metrics for trading experiments."""
    
    # Core RL metrics
    reward: Optional[float] = None
    episode_reward: Optional[float] = None
    loss: Optional[float] = None
    actor_loss: Optional[float] = None
    critic_loss: Optional[float] = None
    epsilon: Optional[float] = None
    learning_rate: Optional[float] = None
    
    # Trading-specific metrics
    portfolio_value: Optional[float] = None
    total_return: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    total_trades: Optional[int] = None
    avg_trade_return: Optional[float] = None
    profit_factor: Optional[float] = None
    
    # System metrics
    training_speed: Optional[float] = None  # steps/second
    memory_usage: Optional[float] = None    # MB
    gpu_utilization: Optional[float] = None # percentage
    
    # Custom metrics (for extensibility)
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary for logging."""
        result = {}
        
        # Add all non-None standard metrics
        for field_name in [
            'reward', 'episode_reward', 'loss', 'actor_loss', 'critic_loss',
            'epsilon', 'learning_rate', 'portfolio_value', 'total_return',
            'sharpe_ratio', 'max_drawdown', 'win_rate', 'total_trades',
            'avg_trade_return', 'profit_factor', 'training_speed',
            'memory_usage', 'gpu_utilization'
        ]:
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = float(value)
        
        # Add custom metrics
        result.update(self.custom_metrics)
        
        return result

@dataclass
class ModelArtifact:
    """Generic model artifact representation."""
    
    name: str
    artifact_type: ArtifactType
    file_path: Optional[str] = None
    content: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    size_bytes: Optional[int] = None
    created_at: Optional[datetime] = None

@dataclass
class ExperimentConfig:
    """Generic experiment configuration."""
    
    # Required fields
    experiment_id: str
    agent_type: str
    
    # Trading parameters
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    initial_balance: Optional[float] = None
    
    # Training parameters
    timesteps: Optional[int] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    
    # Environment parameters
    historical_days: Optional[int] = None
    
    # Custom configuration (for extensibility)
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary (e.g., from training service config)."""
        return cls(
            experiment_id=data.get('experiment_id', ''),
            agent_type=data.get('agent_type', ''),
            symbol=data.get('symbol'),
            timeframe=data.get('timeframe'),
            initial_balance=data.get('initial_balance'),
            timesteps=data.get('timesteps'),
            learning_rate=data.get('learning_rate'),
            batch_size=data.get('batch_size'),
            historical_days=data.get('historical_days'),
            custom_config={k: v for k, v in data.items() if k not in [
                'experiment_id', 'agent_type', 'symbol', 'timeframe',
                'initial_balance', 'timesteps', 'learning_rate', 
                'batch_size', 'historical_days'
            ]},
            created_at=datetime.now()
        )

@dataclass
class ExperimentSummary:
    """Complete experiment summary with metrics and metadata."""
    
    # Basic identification
    experiment_id: str
    run_id: Optional[str] = None
    name: Optional[str] = None
    
    # Status and timing
    status: ExperimentStatus = ExperimentStatus.STARTING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Configuration summary
    agent_type: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    timesteps: Optional[int] = None
    
    # Progress tracking
    current_step: int = 0
    total_steps: int = 0
    progress: float = 0.0
    
    # Key performance metrics
    final_metrics: Dict[str, float] = field(default_factory=dict)
    best_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Detailed metric history (for charts)
    metric_history: Dict[str, List[float]] = field(default_factory=dict)
    metric_steps: Dict[str, List[int]] = field(default_factory=dict)
    
    # Model artifacts
    artifacts: List[ModelArtifact] = field(default_factory=list)
    
    # Configuration and metadata
    config: Optional[ExperimentConfig] = None
    tags: Dict[str, str] = field(default_factory=dict)
    
    # System info
    created_by: Optional[str] = None
    backend_name: Optional[str] = None  # "aim", "wandb", etc.
    
    @property
    def is_running(self) -> bool:
        """Check if experiment is currently running."""
        return self.status in [ExperimentStatus.STARTING, ExperimentStatus.RUNNING]
    
    @property
    def is_completed(self) -> bool:
        """Check if experiment completed successfully."""
        return self.status == ExperimentStatus.COMPLETED
    
    @property
    def has_metrics(self) -> bool:
        """Check if experiment has any performance metrics."""
        return bool(self.final_metrics or self.best_metrics)

@dataclass
class SystemHealth:
    """System health status for tracking backend."""
    
    backend_name: str
    is_healthy: bool
    connected_components: Dict[str, bool] = field(default_factory=dict)
    error_message: Optional[str] = None
    last_check: Optional[datetime] = None
    
    # Component-specific health
    tracker_healthy: bool = True
    repository_healthy: bool = True
    storage_healthy: bool = True
    
    # Statistics
    total_experiments: int = 0
    active_experiments: int = 0
    total_runs: int = 0
    storage_size_mb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'backend_name': self.backend_name,
            'is_healthy': self.is_healthy,
            'connected_components': self.connected_components,
            'error_message': self.error_message,
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'tracker_healthy': self.tracker_healthy,
            'repository_healthy': self.repository_healthy,
            'storage_healthy': self.storage_healthy,
            'stats': {
                'total_experiments': self.total_experiments,
                'active_experiments': self.active_experiments,
                'total_runs': self.total_runs,
                'storage_size_mb': self.storage_size_mb
            }
        } 