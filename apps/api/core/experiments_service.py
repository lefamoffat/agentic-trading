# NOTE: moved from src/experiments/service.py
"""Unified Experiment Service.

Combines ML tracking (long-term experiment analysis) with real-time messaging
(live status updates) to provide a complete experiment management solution.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.tracking import get_experiment_repository
from src.messaging.factory import get_message_broker
from src.utils.logger import get_logger
from src.types.experiments import Experiment

logger = get_logger(__name__)


class UnifiedExperimentService:  # noqa: WPS230 (large class kept as-is)
    """Combine Aim tracking + Redis messaging into one read model."""

    def __init__(self) -> None:
        self.tracking_repository = None
        self.message_broker = None
        self.tracking_connected = False
        self.messaging_connected = False

    # ------------------------------------------------------------------
    # Initialise back-ends
    # ------------------------------------------------------------------

    async def initialize(self) -> None:  # noqa: WPS231
        """Connect to Aim repository and Redis broker."""
        # Tracking
        try:
            self.tracking_repository = await get_experiment_repository()
            health = await self.tracking_repository.get_system_health()
            self.tracking_connected = health.is_healthy
            logger.info(
                "Tracking backend %s – healthy=%s", health.backend_name, health.is_healthy,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tracking backend connection failed: %s", exc)

        # Messaging
        try:
            self.message_broker = get_message_broker()
            self.messaging_connected = await self.message_broker.health_check()
            logger.info("Messaging backend connected=%s", self.messaging_connected)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Messaging backend connection failed: %s", exc)

    # ------------------------------------------------------------------
    # Summary helpers (same implementation as before)
    # ------------------------------------------------------------------

    async def get_experiments_summary(self) -> Dict[str, Any]:  # noqa: WPS231
        """Return combined KPI summary using tracking and messaging back ends."""
        summary: Dict[str, Any] = {
            "total_experiments": 0,
            "total_runs": 0,
            "active_runs": 0,
            "unique_agents": [],
            "unique_symbols": [],
            "backend_name": "unified",
            "backend_healthy": self.tracking_connected or self.messaging_connected,
            "tracking_connected": self.tracking_connected,
            "messaging_connected": self.messaging_connected,
            "recent_experiments": [],
        }

        # Tracking backend metrics
        if self.tracking_connected and self.tracking_repository:
            try:
                health = await self.tracking_repository.get_system_health()
                tracking_experiments = await self.tracking_repository.get_recent_experiments(limit=100)

                summary["total_experiments"] = health.total_experiments
                summary["total_runs"] = health.total_runs

                agents = {exp.agent_type for exp in tracking_experiments if exp.agent_type}
                symbols = {exp.symbol for exp in tracking_experiments if exp.symbol}

                summary["unique_agents"].extend(agents)
                summary["unique_symbols"].extend(symbols)

                summary["recent_experiments"].extend([
                    {
                        "name": exp.name or exp.experiment_id,
                        "experiment_id": exp.experiment_id,
                        "status": exp.status.value,
                        "start_time": exp.start_time.timestamp() * 1000 if exp.start_time else None,
                        "agent_type": exp.agent_type,
                        "symbol": exp.symbol,
                        "source": "tracking",
                    }
                    for exp in tracking_experiments[:5]
                ])

            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to get tracking summary: %s", exc)

        # Messaging backend – active runs
        if self.messaging_connected and self.message_broker:
            try:
                active_experiments: List[Experiment] = await self.message_broker.list_experiments(status_filter="running")
                summary["active_runs"] = len(active_experiments)

                agents = {exp.config.agent_type for exp in active_experiments}
                symbols = {exp.config.symbol for exp in active_experiments}

                summary["unique_agents"].extend([a for a in agents if a])
                summary["unique_symbols"].extend([s for s in symbols if s])

            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to get messaging summary: %s", exc)

        # Remove duplicates & sort
        summary["unique_agents"] = sorted(set(summary["unique_agents"]))
        summary["unique_symbols"] = sorted(set(summary["unique_symbols"]))

        return summary

    # ------------------------------------------------------------------
    # High-level query APIs (copied 1-for-1)
    # ------------------------------------------------------------------

    async def get_recent_experiments(self, limit: int = 10) -> List[Dict[str, Any]]:  # noqa: WPS231
        experiments: List[Dict[str, Any]] = []

        # Historical experiments from tracking backend
        if self.tracking_connected and self.tracking_repository:
            try:
                tracking_experiments = await self.tracking_repository.get_recent_experiments(limit=limit)

                for exp in tracking_experiments:
                    doc = exp.to_dict()
                    # Ensure nested config exists (older runs might lack it)
                    if "config" not in doc or not doc["config"]:
                        doc["config"] = {
                            "symbol": doc.get("symbol", ""),
                            "agent_type": doc.get("agent_type", ""),
                        }

                    doc.update(
                        {
                            "name": getattr(exp, "name", None) or exp.experiment_id,
                            "duration_seconds": getattr(exp, "duration_seconds", None),
                            "backend_name": getattr(exp, "backend_name", "tracking"),
                            "source": "tracking",
                            "latest_run": {
                                "run_id": getattr(exp, "run_id", None),
                                "status": exp.status.value,
                                "start_time": exp.start_time.timestamp() * 1000 if exp.start_time else None,
                                "end_time": exp.end_time.timestamp() * 1000 if exp.end_time else None,
                            } if getattr(exp, "run_id", None) else None,
                        }
                    )

                    experiments.append(doc)

            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to get tracking experiments: %s", exc)

        # Active experiments from messaging backend
        if self.messaging_connected and self.message_broker:
            try:
                messaging_experiments: List[Experiment] = await self.message_broker.list_experiments()

                for exp in messaging_experiments:
                    experiments.append(exp.to_dict() | {"backend_name": "messaging", "source": "messaging"})

            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to get messaging experiments: %s", exc)

        experiments.sort(key=lambda x: x.get("start_time", 0) or 0, reverse=True)
        return experiments[:limit]

    async def get_experiment_details(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        experiment_data: Optional[Dict[str, Any]] = None

        # Historical details first
        if self.tracking_connected and self.tracking_repository:
            try:
                tracking_data = await self.tracking_repository.get_experiment_details(experiment_id)
                if tracking_data:
                    experiment_data = {
                        "experiment_id": tracking_data.experiment_id,
                        "name": tracking_data.name or tracking_data.experiment_id,
                        "status": tracking_data.status.value,
                        "start_time": tracking_data.start_time.timestamp() * 1000 if tracking_data.start_time else None,
                        "end_time": tracking_data.end_time.timestamp() * 1000 if tracking_data.end_time else None,
                        "duration_seconds": tracking_data.duration_seconds,
                        "agent_type": tracking_data.agent_type,
                        "symbol": tracking_data.symbol,
                        "timeframe": tracking_data.timeframe,
                        "progress": tracking_data.progress,
                        "current_step": tracking_data.current_step,
                        "total_steps": tracking_data.total_steps,
                        "metrics": tracking_data.final_metrics or {},
                        "config": (
                            tracking_data.config.to_dict()
                            if tracking_data.config
                            else {
                                "symbol": tracking_data.symbol,
                                "agent_type": tracking_data.agent_type,
                            }
                        ),
                        "source": "tracking",
                    }
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to get experiment from tracking: %s", exc)

        # Real-time additions
        if self.messaging_connected and self.message_broker:
            try:
                messaging_experiment = await self.message_broker.get_experiment(experiment_id)
                if messaging_experiment:
                    if experiment_data:
                        experiment_data["current_step"] = messaging_experiment.state.current_step
                        experiment_data["status"] = messaging_experiment.state.status.value
                        if messaging_experiment.state.total_steps:
                            experiment_data["progress"] = (
                                messaging_experiment.state.current_step / messaging_experiment.state.total_steps
                            )
                        if messaging_experiment.state.metrics:
                            experiment_data["metrics"].update(messaging_experiment.state.metrics)
                    else:
                        experiment_data = messaging_experiment.to_dict() | {"source": "messaging"}
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to get experiment from messaging: %s", exc)

        return experiment_data

    async def health_check(self) -> Dict[str, Any]:
        health: Dict[str, Any] = {
            "overall_healthy": False,
            "tracking_backend": {
                "connected": self.tracking_connected,
                "healthy": False,
                "backend_name": None,
                "error": None,
            },
            "messaging_backend": {
                "connected": self.messaging_connected,
                "healthy": False,
                "error": None,
            },
        }

        # Tracking health
        if self.tracking_connected and self.tracking_repository:
            try:
                tracking_health = await self.tracking_repository.get_system_health()
                health["tracking_backend"].update({
                    "healthy": tracking_health.is_healthy,
                    "backend_name": tracking_health.backend_name,
                    "error": tracking_health.error_message,
                    "total_experiments": tracking_health.total_experiments,
                    "total_runs": tracking_health.total_runs,
                })
            except Exception as exc:  # noqa: BLE001
                health["tracking_backend"]["error"] = str(exc)

        # Messaging health
        if self.messaging_connected and self.message_broker:
            try:
                messaging_healthy = await self.message_broker.health_check()
                health["messaging_backend"]["healthy"] = messaging_healthy
            except Exception as exc:  # noqa: BLE001
                health["messaging_backend"]["error"] = str(exc)

        health["overall_healthy"] = health["tracking_backend"]["healthy"] or health["messaging_backend"]["healthy"]
        return health 