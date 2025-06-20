#!/usr/bin/env python3
"""Reasoning tracker for LLM decision making in trading experiments.

This module tracks, stores, and provides analysis of LLM reasoning
to ensure full auditability and explainability of AI decisions.
"""
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

# Optional imports
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


@dataclass
class ReasoningLog:
    """Structured representation of LLM reasoning for a decision."""
    timestamp: str
    experiment_id: Optional[str]
    action: str  # start, continue, optimize
    llm_model: str
    input_data_summary: Dict[str, Any]
    llm_reasoning: str
    recommended_changes: Dict[str, Any]
    confidence_score: float
    fallback_used: bool
    execution_time_ms: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningLog":
        """Create from dictionary."""
        return cls(**data)


class ReasoningTracker:
    """Tracks and manages LLM reasoning for trading decisions."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize reasoning tracker.
        
        Args:
            storage_path: Optional path for local reasoning storage
        """
        self.logger = get_logger(self.__class__.__name__)
        self.storage_path = storage_path or Path("logs/reasoning")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Current reasoning context
        self._current_reasoning: Optional[ReasoningLog] = None
        self._start_time: Optional[float] = None
    
    def start_reasoning_session(
        self, 
        action: str, 
        experiment_id: Optional[str] = None,
        llm_model: str = "gpt-4"
    ) -> None:
        """Start a new reasoning session.
        
        Args:
            action: Type of action (start, continue, optimize)
            experiment_id: Optional experiment ID for continuation
            llm_model: LLM model being used
        """
        self._start_time = time.time()
        self._current_reasoning = ReasoningLog(
            timestamp=datetime.now().isoformat(),
            experiment_id=experiment_id,
            action=action,
            llm_model=llm_model,
            input_data_summary={},
            llm_reasoning="",
            recommended_changes={},
            confidence_score=0.0,
            fallback_used=False,
            execution_time_ms=0
        )
        
        self.logger.info(f"Started reasoning session: {action} (experiment: {experiment_id})")
    
    def log_input_data(self, data_summary: Dict[str, Any]) -> None:
        """Log the input data that informed the LLM decision.
        
        Args:
            data_summary: Summary of input data used for reasoning
        """
        if self._current_reasoning is None:
            raise ValueError("No active reasoning session. Call start_reasoning_session() first.")
        
        self._current_reasoning.input_data_summary = data_summary
        self.logger.debug(f"Logged input data: {list(data_summary.keys())}")
    
    def log_llm_reasoning(
        self, 
        reasoning: str, 
        confidence_score: float,
        fallback_used: bool = False
    ) -> None:
        """Log the LLM's reasoning and confidence.
        
        Args:
            reasoning: The LLM's textual reasoning
            confidence_score: Confidence score (0.0 to 1.0)
            fallback_used: Whether fallback logic was used
        """
        if self._current_reasoning is None:
            raise ValueError("No active reasoning session. Call start_reasoning_session() first.")
        
        self._current_reasoning.llm_reasoning = reasoning
        self._current_reasoning.confidence_score = confidence_score
        self._current_reasoning.fallback_used = fallback_used
        
        self.logger.info(f"LLM reasoning logged (confidence: {confidence_score:.2f})")
    
    def log_recommended_changes(self, changes: Dict[str, Any]) -> None:
        """Log the recommended configuration changes.
        
        Args:
            changes: Dictionary of recommended changes
        """
        if self._current_reasoning is None:
            raise ValueError("No active reasoning session. Call start_reasoning_session() first.")
        
        self._current_reasoning.recommended_changes = changes
        self.logger.info(f"Recommended changes logged: {list(changes.keys())}")
    
    def finalize_reasoning_session(self) -> ReasoningLog:
        """Finalize the current reasoning session and store it.
        
        Returns:
            The completed reasoning log
        """
        if self._current_reasoning is None:
            raise ValueError("No active reasoning session to finalize.")
        
        if self._start_time is not None:
            execution_time = (time.time() - self._start_time) * 1000
            self._current_reasoning.execution_time_ms = int(execution_time)
        
        # Store locally
        self._store_reasoning_log(self._current_reasoning)
        
        # Store in MLflow if available
        self._store_in_mlflow(self._current_reasoning)
        
        completed_log = self._current_reasoning
        self._current_reasoning = None
        self._start_time = None
        
        self.logger.info(f"Reasoning session finalized (ID: {completed_log.experiment_id})")
        return completed_log
    
    def _store_reasoning_log(self, reasoning_log: ReasoningLog) -> None:
        """Store reasoning log locally.
        
        Args:
            reasoning_log: The reasoning log to store
        """
        filename = f"reasoning_{reasoning_log.action}_{reasoning_log.timestamp.replace(':', '-')}.json"
        file_path = self.storage_path / filename
        
        with open(file_path, 'w') as f:
            json.dump(reasoning_log.to_dict(), f, indent=2)
        
        self.logger.debug(f"Reasoning log stored: {file_path}")
    
    def _store_in_mlflow(self, reasoning_log: ReasoningLog) -> None:
        """Store reasoning in MLflow if available.
        
        Args:
            reasoning_log: The reasoning log to store
        """
        if not MLFLOW_AVAILABLE:
            self.logger.debug("MLflow not available, skipping MLflow storage")
            return
            
        try:
            # Store as tags for searchability
            mlflow.set_tags({
                "llm_action": reasoning_log.action,
                "llm_model": reasoning_log.llm_model,
                "llm_confidence": f"{reasoning_log.confidence_score:.2f}",
                "llm_fallback": str(reasoning_log.fallback_used),
            })
            
            # Store full reasoning as artifact
            reasoning_file = f"reasoning_{reasoning_log.timestamp.replace(':', '-')}.json"
            with open(reasoning_file, 'w') as f:
                json.dump(reasoning_log.to_dict(), f, indent=2)
            
            mlflow.log_artifact(reasoning_file, "reasoning")
            Path(reasoning_file).unlink()  # Clean up temp file
            
            self.logger.debug("Reasoning stored in MLflow")
            
        except Exception as e:
            self.logger.warning(f"Failed to store reasoning in MLflow: {e}")
    
    def get_recent_reasoning(
        self, 
        days: int = 30, 
        action: Optional[str] = None
    ) -> List[ReasoningLog]:
        """Get recent reasoning logs.
        
        Args:
            days: Number of days to look back
            action: Optional action filter (start, continue, optimize)
            
        Returns:
            List of reasoning logs
        """
        reasoning_logs = []
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        for file_path in self.storage_path.glob("reasoning_*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                reasoning_log = ReasoningLog.from_dict(data)
                log_time = datetime.fromisoformat(reasoning_log.timestamp).timestamp()
                
                if log_time >= cutoff_time:
                    if action is None or reasoning_log.action == action:
                        reasoning_logs.append(reasoning_log)
                        
            except Exception as e:
                self.logger.error(f"Failed to load reasoning log {file_path}: {e}")
        
        return sorted(reasoning_logs, key=lambda x: x.timestamp, reverse=True)
    
    def analyze_reasoning_performance(self, symbol: str = "EUR/USD") -> Dict[str, Any]:
        """Analyze correlation between LLM reasoning and performance.
        
        Args:
            symbol: Trading symbol to analyze
            
        Returns:
            Performance analysis data
        """
        # This would integrate with MLflow to correlate reasoning with actual performance
        # For now, return a placeholder structure
        
        recent_logs = self.get_recent_reasoning(days=30)
        
        analysis_data = []
        for log in recent_logs:
            analysis_data.append({
                "timestamp": log.timestamp,
                "action": log.action,
                "confidence": log.confidence_score,
                "changes_count": len(log.recommended_changes),
                "execution_time": log.execution_time_ms,
                "fallback_used": log.fallback_used,
                # TODO: Add actual performance metrics from MLflow
                "performance_improvement": None  # Placeholder
            })
        
        if PANDAS_AVAILABLE:
            return pd.DataFrame(analysis_data).to_dict()
        else:
            return {"raw_data": analysis_data, "note": "pandas not available"}
    
    def export_reasoning_summary(self, file_path: Path) -> None:
        """Export reasoning summary for analysis.
        
        Args:
            file_path: Path to export the summary
        """
        recent_logs = self.get_recent_reasoning(days=30)
        
        summary = {
            "total_decisions": len(recent_logs),
            "actions_breakdown": {},
            "average_confidence": 0.0,
            "fallback_usage": 0.0,
            "recent_logs": [log.to_dict() for log in recent_logs[:10]]  # Last 10
        }
        
        if recent_logs:
            # Calculate statistics
            actions = [log.action for log in recent_logs]
            summary["actions_breakdown"] = {action: actions.count(action) for action in set(actions)}
            summary["average_confidence"] = sum(log.confidence_score for log in recent_logs) / len(recent_logs)
            summary["fallback_usage"] = sum(1 for log in recent_logs if log.fallback_used) / len(recent_logs)
        
        with open(file_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Reasoning summary exported to {file_path}") 