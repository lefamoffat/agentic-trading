"""MLflow Service.

Service for connecting to MLflow tracking server and retrieving experiment data.
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLflowService:
    """Service for interacting with MLflow tracking server."""
    
    def __init__(self):
        """Initialize MLflow service."""
        self.client = None
        self.connected = False
        
        if MLFLOW_AVAILABLE:
            self._connect()
    
    def _connect(self) -> bool:
        """Connect to MLflow tracking server."""
        try:
            # Set MLflow tracking URI if provided
            tracking_uri = os.getenv('MLFLOW_TRACKING_URI', './mlruns')
            mlflow.set_tracking_uri(tracking_uri)
            
            self.client = MlflowClient()
            
            # Test connection by listing experiments
            experiments = self.client.search_experiments()
            self.connected = True
            
            logger.info(f"Connected to MLflow at {tracking_uri}")
            logger.info(f"Found {len(experiments)} experiments")
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to connect to MLflow: {e}")
            self.connected = False
            return False
    
    def get_recent_experiments(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent experiments from MLflow.
        
        Args:
            limit: Maximum number of experiments to return
            
        Returns:
            List of experiment data dictionaries
        """
        if not self.connected:
            logger.warning("MLflow not connected, returning empty list")
            return []
        
        try:
            # Get all experiments
            experiments = self.client.search_experiments()
            
            all_runs = []
            for experiment in experiments:
                try:
                    runs = self.client.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        max_results=limit,
                        order_by=["start_time DESC"]
                    )
                    all_runs.extend(runs)
                except Exception as e:
                    logger.warning(f"Failed to get runs for experiment {experiment.name}: {e}")
                    continue
            
            # Sort all runs by start time and take the most recent
            all_runs.sort(key=lambda x: x.info.start_time or 0, reverse=True)
            recent_runs = all_runs[:limit]
            
            results = []
            for run in recent_runs:
                try:
                    run_data = self._format_run_data(run)
                    results.append(run_data)
                except Exception as e:
                    logger.warning(f"Failed to format run {run.info.run_id}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get recent experiments: {e}")
            return []
    
    def get_experiment_by_id(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific experiment by ID.
        
        Args:
            experiment_id: MLflow experiment ID
            
        Returns:
            Experiment data dictionary or None
        """
        if not self.connected:
            return None
        
        try:
            run = self.client.get_run(experiment_id)
            return self._format_run_data(run)
            
        except Exception as e:
            logger.error(f"Failed to get experiment {experiment_id}: {e}")
            return None
    
    def get_top_models(self, limit: int = 10, metric: str = "eval/sharpe_ratio") -> List[Dict[str, Any]]:
        """
        Get top performing models based on a metric.
        
        Args:
            limit: Maximum number of models to return
            metric: Metric to sort by
            
        Returns:
            List of top model data
        """
        if not self.connected:
            return []
        
        try:
            # Get all experiments
            experiments = self.client.search_experiments()
            
            all_runs = []
            for experiment in experiments:
                try:
                    runs = self.client.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        filter_string=f"metrics.`{metric}` IS NOT NULL",
                        order_by=[f"metrics.`{metric}` DESC"],
                        max_results=limit
                    )
                    all_runs.extend(runs)
                except Exception as e:
                    logger.warning(f"Failed to get runs for experiment {experiment.name}: {e}")
                    continue
            
            # Sort by the metric and take top performers
            valid_runs = [run for run in all_runs if metric in run.data.metrics]
            valid_runs.sort(key=lambda x: x.data.metrics.get(metric, 0), reverse=True)
            top_runs = valid_runs[:limit]
            
            results = []
            for run in top_runs:
                try:
                    run_data = self._format_run_data(run)
                    results.append(run_data)
                except Exception as e:
                    logger.warning(f"Failed to format run {run.info.run_id}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get top models: {e}")
            return []
    
    def get_experiments_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about experiments.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.connected:
            return {
                "total_experiments": 0,
                "total_runs": 0,
                "active_runs": 0,
                "unique_symbols": [],
                "unique_agents": []
            }
        
        try:
            experiments = self.client.search_experiments()
            total_experiments = len(experiments)
            
            all_runs = []
            for experiment in experiments:
                try:
                    runs = self.client.search_runs(
                        experiment_ids=[experiment.experiment_id]
                    )
                    all_runs.extend(runs)
                except Exception:
                    continue
            
            total_runs = len(all_runs)
            active_runs = len([run for run in all_runs if run.info.status == "RUNNING"])
            
            # Extract unique symbols and agents from parameters
            symbols = set()
            agents = set()
            
            for run in all_runs:
                if 'symbol' in run.data.params:
                    symbols.add(run.data.params['symbol'])
                if 'agent' in run.data.params:
                    agents.add(run.data.params['agent'])
            
            return {
                "total_experiments": total_experiments,
                "total_runs": total_runs,
                "active_runs": active_runs,
                "unique_symbols": list(symbols),
                "unique_agents": list(agents)
            }
            
        except Exception as e:
            logger.error(f"Failed to get experiments summary: {e}")
            return {
                "total_experiments": 0,
                "total_runs": 0,
                "active_runs": 0,
                "unique_symbols": [],
                "unique_agents": []
            }
    
    def _format_run_data(self, run) -> Dict[str, Any]:
        """Format MLflow run data for dashboard consumption."""
        return {
            'id': run.info.run_id,
            'name': run.data.tags.get('mlflow.runName', f"Run {run.info.run_id[:8]}"),
            'experiment_name': run.data.tags.get('mlflow.experimentName', 'Default'),
            'status': run.info.status,
            'start_time': datetime.fromtimestamp(run.info.start_time / 1000) if run.info.start_time else None,
            'end_time': datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
            
            # Parameters
            'agent': run.data.params.get('agent', 'Unknown'),
            'symbol': run.data.params.get('symbol', 'Unknown'),
            'timeframe': run.data.params.get('timeframe', 'Unknown'),
            'timesteps': run.data.params.get('timesteps', 'Unknown'),
            
            # Metrics
            'sharpe_ratio': run.data.metrics.get('eval/sharpe_ratio'),
            'total_return': run.data.metrics.get('eval/total_return'),
            'max_drawdown': run.data.metrics.get('eval/max_drawdown'),
            'win_rate': run.data.metrics.get('eval/win_rate'),
            'avg_trade_return': run.data.metrics.get('eval/avg_trade_return'),
            
            # Training metrics
            'train_reward': run.data.metrics.get('train/reward'),
            'train_loss': run.data.metrics.get('train/loss'),
            
            # All metrics for detailed view
            'all_metrics': dict(run.data.metrics),
            'all_params': dict(run.data.params),
            'all_tags': dict(run.data.tags)
        }


# Global service instance
mlflow_service = MLflowService() 