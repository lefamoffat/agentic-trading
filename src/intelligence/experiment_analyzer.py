#!/usr/bin/env python3
"""Experiment analyzer for extracting insights from trading experiments.

This module analyzes historical experiment data from the generic ML tracking system
to provide rich context for LLM-based decision making.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.exceptions import ConfigurationError
from src.utils.logger import get_logger

# ML tracking integration
try:
    from src.tracking import get_experiment_repository, get_ml_tracker
    TRACKING_AVAILABLE = True
except ImportError:
    TRACKING_AVAILABLE = False

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
            experiment_name: Experiment name to analyze
        """
        self.logger = get_logger(self.__class__.__name__)
        self.experiment_name = experiment_name
        self.repository = None
        if TRACKING_AVAILABLE:
            import asyncio
            self.repository = asyncio.run(get_experiment_repository())

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
        if not self.repository:
            self.logger.warning("ML tracking repository not available, returning empty experiments list")
            return []
            
        try:
            import asyncio
            experiments = asyncio.run(self.repository.get_recent_experiments(limit=100))
            
            # Filter by date
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_experiments = [
                exp for exp in experiments 
                if exp.start_time and exp.start_time >= cutoff_date
            ]
            
            summaries = []
            for exp in recent_experiments:
                summary = self._create_experiment_summary_from_generic(exp)
                
                # Apply filters
                if symbol and summary.symbol != symbol:
                    continue
                if agent_type and summary.agent_type != agent_type:
                    continue
                    
                summaries.append(summary)
            
            self.logger.info(f"Analyzed {len(summaries)} recent experiments")
            return summaries
            
        except Exception as e:
            self.logger.error(f"Failed to analyze recent experiments: {e}")
            return []
    
    def _create_experiment_summary_from_generic(self, experiment) -> ExperimentSummary:
        """Create experiment summary from generic experiment object.
        
        Args:
            experiment: Generic experiment object
            
        Returns:
            ExperimentSummary object
        """
        return ExperimentSummary(
            experiment_id=experiment.experiment_id,
            run_id=experiment.run_id or experiment.experiment_id,
            experiment_name=experiment.name or self.experiment_name,
            status=experiment.status.value if hasattr(experiment.status, 'value') else str(experiment.status),
            start_time=experiment.start_time,
            end_time=experiment.end_time,
            duration_minutes=experiment.duration_minutes if hasattr(experiment, 'duration_minutes') else None,
            final_sharpe_ratio=experiment.final_metrics.get('sharpe_ratio') if experiment.final_metrics else None,
            final_profit=experiment.final_metrics.get('total_profit') if experiment.final_metrics else None,
            max_drawdown=experiment.final_metrics.get('max_drawdown') if experiment.final_metrics else None,
            total_trades=experiment.final_metrics.get('total_trades') if experiment.final_metrics else None,
            win_rate=experiment.final_metrics.get('win_rate') if experiment.final_metrics else None,
            agent_type=experiment.agent_type,
            symbol=experiment.symbol,
            timeframe=experiment.timeframe,
            initial_balance=experiment.config.initial_balance if experiment.config else 0.0,
            learning_rate=experiment.config.learning_rate if experiment.config and hasattr(experiment.config, 'learning_rate') else None,
            total_timesteps=experiment.timesteps,
            best_episode=None,  # Not available in generic interface
            convergence_episode=None,  # Not available in generic interface
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
        if current_run_id and action in ["continue", "optimize"] and self.repository:
            try:
                import asyncio
                # Try to find the experiment by run ID
                experiments = asyncio.run(self.repository.get_recent_experiments(limit=100))
                current_experiment = None
                for exp in experiments:
                    if exp.run_id == current_run_id or exp.experiment_id == current_run_id:
                        current_experiment = exp
                        break
                
                if current_experiment:
                    summary["current_experiment"] = {
                        "run_id": current_run_id,
                        "status": current_experiment.status.value if hasattr(current_experiment.status, 'value') else str(current_experiment.status),
                        "metrics": current_experiment.final_metrics or {},
                        "params": {
                            "agent_type": current_experiment.agent_type,
                            "symbol": current_experiment.symbol,
                            "timeframe": current_experiment.timeframe,
                            "timesteps": current_experiment.timesteps,
                        },
                    }
                else:
                    summary["current_experiment"] = {
                        "run_id": current_run_id,
                        "note": "Experiment not found in recent experiments"
                    }
            except Exception as e:
                self.logger.error(f"Failed to get current experiment details: {e}")
        elif current_run_id and not self.repository:
            summary["current_experiment"] = {
                "run_id": current_run_id,
                "note": "ML tracking repository not available, cannot fetch experiment details"
            }
        
        return summary 