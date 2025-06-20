#!/usr/bin/env python3
"""Experiment analyzer for extracting insights from trading experiments.

This module analyzes historical experiment data from MLflow, Qlib, and Optuna
to provide rich context for LLM-based decision making.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from mlflow.entities import Experiment, Run

from src.utils.exceptions import ConfigurationError
from src.utils.logger import get_logger

# Optional imports
try:
    import mlflow
    from mlflow.entities import Experiment, Run
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    Experiment = None
    Run = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


@dataclass
class ExperimentSummary:
    """Summary of a trading experiment."""
    experiment_id: str
    run_id: str
    experiment_name: str
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_minutes: Optional[float]
    
    # Performance metrics
    final_sharpe_ratio: Optional[float]
    final_profit: Optional[float] 
    max_drawdown: Optional[float]
    total_trades: Optional[int]
    win_rate: Optional[float]
    
    # Configuration
    agent_type: Optional[str]
    symbol: str
    timeframe: str
    initial_balance: float
    learning_rate: Optional[float]
    
    # Training details
    total_timesteps: Optional[int]
    best_episode: Optional[int]
    convergence_episode: Optional[int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_minutes": self.duration_minutes,
            "final_sharpe_ratio": self.final_sharpe_ratio,
            "final_profit": self.final_profit,
            "max_drawdown": self.max_drawdown,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "agent_type": self.agent_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "initial_balance": self.initial_balance,
            "learning_rate": self.learning_rate,
            "total_timesteps": self.total_timesteps,
            "best_episode": self.best_episode,
            "convergence_episode": self.convergence_episode,
        }


class ExperimentAnalyzer:
    """Analyzes trading experiments to extract insights for LLM reasoning."""
    
    def __init__(self, experiment_name: str = "AgenticTrading"):
        """Initialize experiment analyzer.
        
        Args:
            experiment_name: MLflow experiment name to analyze
        """
        self.logger = get_logger(self.__class__.__name__)
        self.experiment_name = experiment_name
        self.client = mlflow.tracking.MlflowClient()
    
    def get_experiment_by_name(self, name: str) -> Optional[Experiment]:
        """Get experiment by name.
        
        Args:
            name: Experiment name
            
        Returns:
            Experiment object or None if not found
        """
        try:
            return self.client.get_experiment_by_name(name)
        except Exception as e:
            self.logger.warning(f"Experiment '{name}' not found: {e}")
            return None
    
    def analyze_recent_experiments(
        self, 
        days: int = 30,
        symbol: Optional[str] = None,
        agent_type: Optional[str] = None
    ) -> List[ExperimentSummary]:
        """Analyze recent experiments for patterns and insights.
        
        Args:
            days: Number of days to look back
            symbol: Optional symbol filter
            agent_type: Optional agent type filter
            
        Returns:
            List of experiment summaries
        """
        if not self.client:
            self.logger.warning("MLflow client not available, returning empty experiments list")
            return []
            
        experiment = self.get_experiment_by_name(self.experiment_name)
        if not experiment:
            self.logger.warning(f"No experiment found with name: {self.experiment_name}")
            return []
        
        # Get recent runs
        cutoff_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"attribute.start_time > {cutoff_time}",
            order_by=["attribute.start_time DESC"],
            max_results=100
        )
        
        summaries = []
        for run in runs:
            summary = self._create_experiment_summary(run, experiment.experiment_id)
            
            # Apply filters
            if symbol and summary.symbol != symbol:
                continue
            if agent_type and summary.agent_type != agent_type:
                continue
                
            summaries.append(summary)
        
        self.logger.info(f"Analyzed {len(summaries)} recent experiments")
        return summaries
    
    def _create_experiment_summary(self, run: Run, experiment_id: str) -> ExperimentSummary:
        """Create experiment summary from MLflow run.
        
        Args:
            run: MLflow run object
            experiment_id: Experiment ID
            
        Returns:
            ExperimentSummary object
        """
        info = run.info
        data = run.data
        
        # Parse timestamps
        start_time = datetime.fromtimestamp(info.start_time / 1000) if info.start_time else None
        end_time = datetime.fromtimestamp(info.end_time / 1000) if info.end_time else None
        
        duration_minutes = None
        if start_time and end_time:
            duration_minutes = (end_time - start_time).total_seconds() / 60
        
        # Extract metrics (with safe access)
        metrics = data.metrics
        final_sharpe_ratio = metrics.get("eval/sharpe_ratio")
        final_profit = metrics.get("eval/total_profit")
        max_drawdown = metrics.get("eval/max_drawdown")
        total_trades = metrics.get("eval/total_trades")
        win_rate = metrics.get("eval/win_rate")
        
        # Extract parameters
        params = data.params
        agent_type = params.get("agent", "unknown")
        symbol = params.get("symbol", "unknown")
        timeframe = params.get("timeframe", "unknown")
        initial_balance = float(params.get("initial_balance", 0))
        learning_rate = float(params.get("learning_rate", 0)) if params.get("learning_rate") else None
        total_timesteps = int(params.get("timesteps", 0)) if params.get("timesteps") else None
        
        # Extract training details (these might be in metrics)
        best_episode = metrics.get("best_episode")
        convergence_episode = metrics.get("convergence_episode")
        
        return ExperimentSummary(
            experiment_id=experiment_id,
            run_id=info.run_id,
            experiment_name=self.experiment_name,
            status=info.status,
            start_time=start_time,
            end_time=end_time,
            duration_minutes=duration_minutes,
            final_sharpe_ratio=final_sharpe_ratio,
            final_profit=final_profit,
            max_drawdown=max_drawdown,
            total_trades=total_trades,
            win_rate=win_rate,
            agent_type=agent_type,
            symbol=symbol,
            timeframe=timeframe,
            initial_balance=initial_balance,
            learning_rate=learning_rate,
            total_timesteps=total_timesteps,
            best_episode=best_episode,
            convergence_episode=convergence_episode,
        )
    
    def summarize_for_llm(
        self, 
        action: str,
        current_run_id: Optional[str] = None,
        symbol: str = "EUR/USD",
        days: int = 30
    ) -> Dict[str, Any]:
        """Create a comprehensive summary for LLM reasoning.
        
        Args:
            action: Type of action (start, continue, optimize)
            current_run_id: Current experiment run ID (for continue/optimize)
            symbol: Trading symbol
            days: Days to look back for analysis
            
        Returns:
            Comprehensive summary for LLM
        """
        summary = {
            "action_type": action,
            "analysis_timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "lookback_days": days,
        }
        
        # Get recent experiments
        recent_experiments = self.analyze_recent_experiments(
            days=days, symbol=symbol
        )
        
        if recent_experiments:
            summary["historical_context"] = {
                "total_experiments": len(recent_experiments),
                "recent_performance": [exp.to_dict() for exp in recent_experiments[:5]],
            }
        else:
            summary["historical_context"] = {"note": "No recent experiments found"}
        
        # Add current experiment analysis for continue/optimize actions
        if current_run_id and action in ["continue", "optimize"] and self.client:
            try:
                current_run = self.client.get_run(current_run_id)
                summary["current_experiment"] = {
                    "run_id": current_run_id,
                    "status": current_run.info.status,
                    "metrics": current_run.data.metrics,
                    "params": current_run.data.params,
                }
            except Exception as e:
                self.logger.error(f"Failed to get current experiment details: {e}")
        elif current_run_id and not self.client:
            summary["current_experiment"] = {
                "run_id": current_run_id,
                "note": "MLflow not available, cannot fetch experiment details"
            }
        
        return summary 