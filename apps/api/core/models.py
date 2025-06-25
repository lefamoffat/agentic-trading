from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class UnifiedExperimentSummary:
    """Unified experiment summary combining tracking and messaging data."""

    experiment_id: str
    name: str
    status: str
    agent_type: str
    symbol: str
    timeframe: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    current_step: int
    total_steps: int
    progress: float
    metrics: Dict[str, float]
    config: Optional[Dict[str, Any]] = None

    # Data source indicators
    has_tracking_data: bool = False
    has_messaging_data: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "status": self.status,
            "agent_type": self.agent_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress": self.progress,
            "metrics": self.metrics,
            "config": self.config,
            "has_tracking_data": self.has_tracking_data,
            "has_messaging_data": self.has_messaging_data,
        } 