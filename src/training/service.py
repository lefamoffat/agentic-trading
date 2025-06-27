"""Training service for background experiment execution."""

import asyncio
import time
import traceback
from typing import Dict, Any, List, Optional, Set

from src.messaging import TrainingStatus
from src.messaging.factory import get_message_broker
from src.messaging.channels.training import TrainingChannel
from src.tracking import get_ml_tracker
from src.utils.logger import get_logger

# Import modular components
from .config_validator import validate_config
from .event_publisher import EventPublisher
from .training_executor import TrainingExecutor

# Event loop-aware utilities
from src.training.utils import async_guard

logger = get_logger(__name__)

class TrainingService:
    """Orchestrates training workflows using modular components."""
    
    def __init__(self, loop=None):
        """Initialize training service."""
        self.training_channel = None
        self.ml_tracker = None
        self.event_publisher = None
        self.training_executor = None
        self._running = False
        self._active_experiments: Set[str] = set()
        self.loop = loop
        
    async def start(self) -> None:
        """Start the training service."""
        if self._running:
            return
            
        try:
            logger.info("[TRAINING] Starting training service")
            
            # Get message broker and create training channel
            logger.info("[TRAINING] Getting message broker")
            message_broker = get_message_broker()
            
            # Test broker connection
            logger.info("[TRAINING] Testing broker connection")
            try:
                if not await message_broker.health_check():
                    raise RuntimeError("Message broker health check failed")
            except Exception as e:
                logger.error(f"[TRAINING] Message broker health check failed: {e}")
                logger.error(f"[TRAINING] Exception type: {type(e).__name__}")
                logger.error(f"[TRAINING] Traceback: {traceback.format_exc()}")
                raise RuntimeError("Message broker health check failed") from e
            
            logger.info("[TRAINING] Creating training channel")
            self.training_channel = TrainingChannel(message_broker)
            
            # Event publisher depends only on the broker and must exist before the executor
            logger.info("[TRAINING] Creating event publisher")
            self.event_publisher = EventPublisher(self.training_channel.broker)
            
            # Initialize ML tracker
            try:
                logger.info("[TRAINING] Initializing ML tracker")
                self.ml_tracker = await get_ml_tracker()
            except Exception as e:
                logger.error(f"[TRAINING] Failed to initialize ML tracker: {e}")
                logger.error(f"[TRAINING] Exception type: {type(e).__name__}")
                logger.error(f"[TRAINING] Traceback: {traceback.format_exc()}")
                raise
            
            # Initialize training executor (requires event_publisher & ml_tracker)
            logger.info("[TRAINING] Creating training executor")
            self.training_executor = TrainingExecutor(
                self.training_channel,
                self.ml_tracker,
                self.event_publisher,
                loop=self.loop
            )
            
            # Clean up any stale experiments left from previous runs
            logger.info("[TRAINING] Cleaning up stale experiments (if any)")
            await self._cleanup_stale_experiments()
            
            self._running = True
            logger.info("[TRAINING] Training service started successfully")
            
        except Exception as e:
            logger.error(f"[TRAINING] Failed to start training service: {e}")
            logger.error(f"[TRAINING] Exception type: {type(e).__name__}")
            logger.error(f"[TRAINING] Traceback: {traceback.format_exc()}")
            
            # Clean up resources
            if self.training_channel:
                await self.training_channel.broker.close()
            if self.ml_tracker:
                await self.ml_tracker.close()
            
            raise
        
    async def stop(self) -> None:
        """Stop the training service."""
        if not self._running:
            return
            
        # Cancel all active experiments
        active_experiments = list(self._active_experiments)
        for exp_id in active_experiments:
            await self.stop_experiment(exp_id)
            
        self._running = False
        logger.info("Training service stopped")
        
    async def start_experiment(
        self,
        experiment_id: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Start a new training experiment.
        
        Args:
            experiment_id: Unique experiment identifier
            config: Training configuration
            
        Returns:
            Dict with start status
        """
        try:
            # Validate configuration using new validator
            validate_config(config)
            
            # Create experiment using the training channel
            await self.training_channel.create_experiment(experiment_id, config)
            
            # Publish status update using new event publisher
            await self.event_publisher.publish_status_update(
                experiment_id,
                TrainingStatus.STARTING,
                "Experiment starting..."
            )
            
            # Create task but don't start it yet
            task = asyncio.create_task(
                self.training_executor.execute_training(experiment_id, config),
                name=f"training_{experiment_id}"  # Add name for better debugging
            )
            
            # Add error handler before task starts running
            def handle_task_exception(task):
                try:
                    if task.cancelled():
                        logger.warning(f"Training task was cancelled for {experiment_id}")
                        return
                        
                    if not task.done():
                        logger.error(f"Training task not done for {experiment_id}")
                        return
                        
                    exception = task.exception()
                    if exception:
                        logger.error(f"Background training task failed for {experiment_id}: {exception}")
                        logger.error(f"Exception type: {type(exception).__name__}")
                        logger.error(f"Exception traceback: {traceback.format_exception(type(exception), exception, exception.__traceback__)}")
                        
                        # Get the event loop for async operations
                        try:
                            loop = asyncio.get_running_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                        
                        if loop.is_running():
                            # Schedule error handling and wait for completion
                            future = asyncio.run_coroutine_threadsafe(
                                self._handle_background_task_error(experiment_id, exception),
                                loop
                            )
                            try:
                                future.result(timeout=5.0)  # Wait for error handling to complete
                            except Exception as e:
                                logger.error(f"Failed to handle background task error: {e}")
                    else:
                        # Task completed without exception but also without proper completion
                        logger.error(f"Training task completed without proper finalization for {experiment_id}")
                        
                        # Get the event loop for async operations
                        try:
                            loop = asyncio.get_running_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                        
                        if loop.is_running():
                            error = RuntimeError("Training task completed without proper finalization")
                            # Schedule error handling and wait for completion
                            future = asyncio.run_coroutine_threadsafe(
                                self._handle_background_task_error(experiment_id, error),
                                loop
                            )
                            try:
                                future.result(timeout=5.0)  # Wait for error handling to complete
                            except Exception as e:
                                logger.error(f"Failed to handle background task error: {e}")
                except Exception as e:
                    logger.error(f"Error in task exception handler: {e}")
                    logger.error(f"Handler exception traceback: {traceback.format_exc()}")
                finally:
                    # Always remove from active experiments
                    self._active_experiments.discard(experiment_id)
            
            # Add callback and track task BEFORE returning
            task.add_done_callback(handle_task_exception)
            self._active_experiments.add(experiment_id)
            
            logger.info(f"Started experiment {experiment_id}")
            return {
                "experiment_id": experiment_id,
                "status": TrainingStatus.STARTING.value,
                "message": "Experiment started successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to start experiment {experiment_id}: {e}")
            # Ensure error is published and status is updated
            await asyncio.gather(
                self.event_publisher.publish_error(
                    experiment_id,
                    type(e).__name__,
                    str(e),
                    traceback.format_exc()
                ),
                self.training_channel.publish_status(
                    experiment_id,
                    TrainingStatus.FAILED,
                    str(e)
                )
            )
            raise
            
    async def stop_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Stop a running experiment.
        
        Args:
            experiment_id: Experiment ID to stop
            
        Returns:
            Dict with stop status
        """
        try:
            # Get current experiment state
            experiment = await self.training_channel.get_experiment(experiment_id)
            if not experiment:
                return {
                    "experiment_id": experiment_id,
                    "status": "not_found",
                    "message": "Experiment not found"
                }
            
            # Update status to cancelled
            await self.training_channel.publish_status(
                experiment_id,
                TrainingStatus.CANCELLED,
                "Experiment cancelled by user"
            )
            
            # Remove from active experiments
            self._active_experiments.discard(experiment_id)
            
            logger.info(f"Stopped experiment {experiment_id}")
            
            return {
                "experiment_id": experiment_id,
                "status": TrainingStatus.CANCELLED.value,
                "message": "Experiment stopped successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to stop experiment {experiment_id}: {e}")
            raise
            
    @async_guard
    async def _cleanup_stale_experiments(self) -> None:
        """Clean up any stale experiments from previous runs."""
        try:
            # Get all experiments in starting/running state
            stale_experiments = await self.training_channel.list_experiments(
                status_filter=[TrainingStatus.STARTING.value, TrainingStatus.RUNNING.value]
            )
            
            for exp in stale_experiments:
                exp_id = exp.id
                # Mark as failed with explanation
                await self.training_channel.publish_status(
                    exp_id,
                    TrainingStatus.FAILED,
                    "Experiment terminated due to service restart"
                )
                logger.info(f"Cleaned up stale experiment {exp_id}")
                
        except Exception as e:
            logger.error(f"Failed to cleanup stale experiments: {e}")
            
    async def _handle_background_task_error(self, experiment_id: str, exception: Exception) -> None:
        """Handle background task errors by updating experiment status."""
        try:
            await self.training_channel.publish_status(
                experiment_id,
                TrainingStatus.FAILED,
                f"Background task failed: {str(exception)}"
            )
            
            await self.event_publisher.publish_error(
                experiment_id,
                type(exception).__name__,
                str(exception),
                traceback.format_exc()
            )
            
            # Remove from active experiments
            self._active_experiments.discard(experiment_id)
            
        except Exception as e:
            logger.error(f"Failed to handle background task error for {experiment_id}: {e}")
            
    async def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get the current status of an experiment.
        
        Args:
            experiment_id: Experiment to query
            
        Returns:
            Dict with experiment status
        """
        experiment = await self.training_channel.get_experiment(experiment_id)
        if experiment is None:
            return {
                "experiment_id": experiment_id,
                "status": "not_found",
                "message": "Experiment not found",
            }

        # Duration & progress -----------------------------------------
        start_time = experiment.state.start_time
        end_time = experiment.state.end_time

        duration: float | None
        if start_time is None:
            duration = None
        else:
            if end_time is None:
                duration = time.time() - start_time
            else:
                duration = end_time - start_time

        progress: float = (
            experiment.state.current_step / experiment.state.total_steps
            if experiment.state.total_steps > 0
            else 0.0
        )

        return {
            "experiment_id": experiment.id,
            "status": experiment.state.status.value,
            "current_step": experiment.state.current_step,
            "total_steps": experiment.state.total_steps,
            "progress": progress,
            "duration": duration,
            "metrics": experiment.state.metrics,
            "config": experiment.config.model_dump(mode="python"),
        }
        
    async def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments."""
        if not self.training_channel:
            return []
            
        try:
            experiments = await self.training_channel.list_experiments()
            return experiments
        except Exception as e:
            logger.error(f"Error listing experiments: {e}")
            return []
    
    


async def get_training_service() -> TrainingService:
    """
    Get the global training service instance.
    
    Returns:
        TrainingService instance
    """
    global _training_service_instance

    # Detect current loop
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if _training_service_instance is None:
        _training_service_instance = TrainingService(loop=current_loop)
        await _training_service_instance.start()
    else:
        # Check if the stored instance uses a different (or closed) loop
        svc_loop = _training_service_instance.loop
        if svc_loop is not current_loop and (svc_loop is None or (hasattr(svc_loop, "is_closed") and svc_loop.is_closed())):
            await _training_service_instance.stop()
            _training_service_instance = TrainingService(loop=current_loop)
            await _training_service_instance.start()

    return _training_service_instance

# Global instance
_training_service_instance = None 