"""Training executor for ML training workflows."""

import asyncio
import traceback
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from src.messaging import TrainingStatus
from src.tracking import ExperimentConfig, TrainingMetrics
from src.agents.factory import agent_factory
from src.environment import TradingEnv, load_trading_config
from src.utils.logger import get_logger

from .data_processor import process_training_data
from src.training.callbacks.adapters.sb3 import SB3TrainingCallback
from src.training.utils import async_guard

logger = get_logger(__name__)


class TrainingExecutor:
    """Executes the actual ML training workflow."""
    
    def __init__(self, training_channel, ml_tracker, event_publisher, loop=None):
        """Initialize training executor with dependencies."""
        self.training_channel = training_channel
        self.ml_tracker = ml_tracker
        self.event_publisher = event_publisher
        self._active_tasks = set()
        self.loop = loop
    
    @async_guard
    async def execute_training(self, experiment_id: str, config: Dict[str, Any]) -> None:
        """
        Execute the complete training workflow.
        
        Args:
            experiment_id: Experiment identifier
            config: Training configuration
        """
        ml_run = None
        env = None
        agent = None
        
        try:
            logger.info(f"[EXEC] Starting training execution for {experiment_id}")
            
            # Update status to running
            logger.info(f"[EXEC] Updating status to running for {experiment_id}")
            try:
                await self.training_channel.publish_status(experiment_id, TrainingStatus.RUNNING)
                await self.event_publisher.publish_status_update(
                    experiment_id, TrainingStatus.RUNNING, "Preparing training environment"
                )
            except Exception as e:
                logger.error(f"[EXEC] Failed to update status to running: {e}")
                logger.error(f"[EXEC] Exception type: {type(e).__name__}")
                logger.error(f"[EXEC] Traceback: {traceback.format_exc()}")
                await self.training_channel.publish_error(
                    experiment_id,
                    type(e).__name__,
                    str(e),
                    traceback.format_exc()
                )
                await self.event_publisher.publish_error(
                    experiment_id,
                    type(e).__name__,
                    str(e),
                    traceback.format_exc()
                )
                raise
            
            # Process training data
            try:
                logger.info(f"[EXEC] Processing training data for {experiment_id}")
                data = await self._process_data(experiment_id, config)
                if not isinstance(data, pd.DataFrame) or data.empty:
                    err_msg = f"Training data for {experiment_id} is not a non-empty pandas DataFrame. Got: {type(data)} with shape {getattr(data, 'shape', None)}"
                    logger.error(f"[EXEC] {err_msg}")
                    await self.training_channel.publish_error(
                        experiment_id,
                        "DataError",
                        err_msg,
                        ""
                    )
                    await self.event_publisher.publish_error(
                        experiment_id,
                        "DataError",
                        err_msg,
                        ""
                    )
                    raise ValueError(err_msg)
                logger.info(f"[EXEC] Training data processed for {experiment_id}")
            except Exception as e:
                logger.error(f"[EXEC] Failed to process training data: {e}")
                logger.error(f"[EXEC] Exception type: {type(e).__name__}")
                logger.error(f"[EXEC] Traceback: {traceback.format_exc()}")
                await self.training_channel.publish_error(
                    experiment_id,
                    type(e).__name__,
                    str(e),
                    traceback.format_exc()
                )
                await self.event_publisher.publish_error(
                    experiment_id,
                    type(e).__name__,
                    str(e),
                    traceback.format_exc()
                )
                raise
            
            # Create ML run
            try:
                logger.info(f"[EXEC] Creating ML run for {experiment_id}")
                from src.tracking.models import ExperimentConfig
                exp_config = ExperimentConfig.from_dict({**config, 'experiment_id': experiment_id})
                ml_run = await self.ml_tracker.start_run(experiment_id, exp_config)
                logger.info(f"[EXEC] ML run created for {experiment_id}")
            except Exception as e:
                logger.error(f"[EXEC] Failed to create ML run: {e}")
                logger.error(f"[EXEC] Exception type: {type(e).__name__}")
                logger.error(f"[EXEC] Traceback: {traceback.format_exc()}")
                await self.training_channel.publish_error(
                    experiment_id,
                    type(e).__name__,
                    str(e),
                    traceback.format_exc()
                )
                await self.event_publisher.publish_error(
                    experiment_id,
                    type(e).__name__,
                    str(e),
                    traceback.format_exc()
                )
                raise
            
            # Create environment and agent
            try:
                logger.info(f"[EXEC] Creating environment and agent for {experiment_id}")
                env, agent = await self._prepare_training_components(config, data)
                logger.info(f"[EXEC] Environment and agent created for {experiment_id}")
            except Exception as e:
                logger.error(f"[EXEC] Failed to create environment/agent: {e}")
                logger.error(f"[EXEC] Exception type: {type(e).__name__}")
                logger.error(f"[EXEC] Traceback: {traceback.format_exc()}")
                await self.training_channel.publish_error(
                    experiment_id,
                    type(e).__name__,
                    str(e),
                    traceback.format_exc()
                )
                await self.event_publisher.publish_error(
                    experiment_id,
                    type(e).__name__,
                    str(e),
                    traceback.format_exc()
                )
                raise
            
            # Run training
            try:
                logger.info(f"[EXEC] Starting training for {experiment_id}")
                await self._run_training_loop(experiment_id, config, env, agent, ml_run)
                logger.info(f"[EXEC] Training completed for {experiment_id}")
            except Exception as e:
                logger.error(f"[EXEC] Failed during training: {e}")
                logger.error(f"[EXEC] Exception type: {type(e).__name__}")
                logger.error(f"[EXEC] Traceback: {traceback.format_exc()}")
                await self.training_channel.publish_error(
                    experiment_id,
                    type(e).__name__,
                    str(e),
                    traceback.format_exc()
                )
                await self.event_publisher.publish_error(
                    experiment_id,
                    type(e).__name__,
                    str(e),
                    traceback.format_exc()
                )
                raise
            
            # Update status to completed
            try:
                logger.info(f"[EXEC] Updating status to completed for {experiment_id}")
                await self.training_channel.publish_status(experiment_id, TrainingStatus.COMPLETED)
                await self.event_publisher.publish_status_update(
                    experiment_id, TrainingStatus.COMPLETED, "Training completed successfully"
                )
            except Exception as e:
                logger.error(f"[EXEC] Failed to update status to completed: {e}")
                logger.error(f"[EXEC] Exception type: {type(e).__name__}")
                logger.error(f"[EXEC] Traceback: {traceback.format_exc()}")
                await self.training_channel.publish_error(
                    experiment_id,
                    type(e).__name__,
                    str(e),
                    traceback.format_exc()
                )
                await self.event_publisher.publish_error(
                    experiment_id,
                    type(e).__name__,
                    str(e),
                    traceback.format_exc()
                )
                raise
            
        except Exception as e:
            logger.error(f"[EXEC] Training execution failed for {experiment_id}: {e}")
            logger.error(f"[EXEC] Exception type: {type(e).__name__}")
            logger.error(f"[EXEC] Traceback: {traceback.format_exc()}")
            await self.training_channel.publish_error(
                experiment_id,
                type(e).__name__,
                str(e),
                traceback.format_exc()
            )
            await self.event_publisher.publish_error(
                experiment_id,
                type(e).__name__,
                str(e),
                traceback.format_exc()
            )
            raise
            
        finally:
            # No explicit cleanup needed for ml_run, env, or agent (no .close() method)
            pass
    
    async def _process_data(self, experiment_id: str, config: Dict[str, Any]):
        """Process training data."""
        async def status_callback(message: str):
            logger.info(f"[DATA] Status callback for {experiment_id}: {message}")
            await self.event_publisher.publish_status_update(
                experiment_id, TrainingStatus.RUNNING, message
            )
        
        logger.info(f"[DATA] Starting data processing for {experiment_id} with config: {config}")
        try:
            logger.info(f"[DATA] About to call process_training_data for {experiment_id}")
            features_df = await process_training_data(experiment_id, config, status_callback)
            logger.info(f"[DATA] Data processing completed for {experiment_id}")
            return features_df
        except Exception as e:
            logger.error(f"[DATA] Failed to process training data: {e}")
            logger.error(f"[DATA] Exception type: {type(e).__name__}")
            logger.error(f"[DATA] Traceback: {traceback.format_exc()}")
            
            # Update experiment status to failed
            try:
                await self.training_channel.publish_status(experiment_id, TrainingStatus.FAILED)
                await self.event_publisher.publish_status_update(
                    experiment_id,
                    TrainingStatus.FAILED,
                    str(e)
                )
                await self.training_channel.publish_error(
                    experiment_id,
                    type(e).__name__,
                    str(e),
                    traceback.format_exc()
                )
            except Exception as status_error:
                logger.error(f"[DATA] Failed to update status after error: {status_error}")
                logger.error(f"[DATA] Status error type: {type(status_error).__name__}")
                logger.error(f"[DATA] Status error traceback: {traceback.format_exc()}")
            
            # Re-raise the original error
            raise
    
    async def _start_ml_tracking(self, experiment_id: str, config: Dict[str, Any], 
                                features_df, features_path, qlib_data_dir) -> Optional[Any]:
        """Start ML tracking run if tracker is available."""
        if not self.ml_tracker:
            return None
            
        try:
            experiment_config = ExperimentConfig(
                experiment_id=experiment_id,
                agent_type=config.get("agent_type", "PPO"),
                symbol=config["symbol"],
                timeframe=config["timeframe"],
                timesteps=config.get("timesteps", 10000),
                learning_rate=config.get("learning_rate", 0.0003),
                initial_balance=config.get("initial_balance", 10000.0)
            )
            
            ml_run = await self.ml_tracker.start_run(
                experiment_id, experiment_config, f"Training_{experiment_id[:8]}"
            )
            
            # Log hyperparameters
            hyperparams = {
                "agent": config.get("agent_type", "PPO"),
                "symbol": config["symbol"],
                "timeframe": config["timeframe"],
                "timesteps": config.get("timesteps", 10000),
                "learning_rate": config.get("learning_rate", 0.0003),
                "initial_balance": config.get("initial_balance", 10000.0),
                "features_count": len(features_df),
                "feature_columns": len(features_df.columns),
                "features_file": str(features_path),
                "qlib_data_dir": str(qlib_data_dir)
            }
            await self.ml_tracker.log_hyperparameters(ml_run.id, hyperparams)
            
            logger.info(f"Started ML tracking run: {ml_run.id}")
            return ml_run
            
        except Exception as e:
            logger.warning(f"Failed to start ML tracking run: {e}")
            return None
    
    async def _prepare_training_components(self, config: Dict[str, Any], features_df: pd.DataFrame):
        """Prepare training environment and agent."""
        logger.info(f"Preparing training components with config: {config}")
        from pathlib import Path
        try:
            # Load trading environment config from YAML
            base_env_config = load_trading_config(Path("configs/trading_config.yaml"))
            # Override with any values from config dict
            from dataclasses import replace
            env_config = replace(
                base_env_config,
                initial_balance=config.get("initial_balance", base_env_config.initial_balance),
                max_steps=config.get("timesteps", base_env_config.max_steps),
                observation_features=config.get("observation_features", base_env_config.observation_features),
                action_type=base_env_config.action_type,  # Could be overridden if needed
                reward_system=base_env_config.reward_system,  # Could be overridden if needed
                # Add more fields as needed
            )

            # Create environment
            logger.info("Creating trading environment")
            env = TradingEnv(
                data=features_df,
                config=env_config
            )

            # Create agent
            logger.info(f"Creating agent of type: {config.get('agent_type', 'ppo')}")
            agent = agent_factory.create_agent(
                name=config.get("agent_type", "ppo"),
                env=env,
                hyperparams={"learning_rate": config.get("learning_rate", 0.0003)}
            )

            logger.info("Training components prepared successfully")
            return env, agent
        except Exception as e:
            logger.error(f"Failed to prepare training components: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            await self.training_channel.publish_error(
                config.get("experiment_id", "unknown"),
                type(e).__name__,
                str(e),
                traceback.format_exc()
            )
            await self.event_publisher.publish_error(
                config.get("experiment_id", "unknown"),
                type(e).__name__,
                str(e),
                traceback.format_exc()
            )
            raise
    
    async def _run_training_loop(self, experiment_id: str, config: Dict[str, Any], 
                                env, agent, ml_run) -> None:
        """Execute the actual training loop."""
        total_timesteps = config.get("timesteps", 10000)
        
        # Create callback for real-time updates
        callback = SB3TrainingCallback(
            experiment_id=experiment_id,
            training_channel=self.training_channel,
            total_timesteps=total_timesteps,
            event_publisher=self.event_publisher,
            loop=self.loop,
            ml_tracker=self.ml_tracker,
            ml_run=ml_run
        )
        logger.debug(f"[EXECUTOR] Created callback id={id(callback)}")
        
        # Execute training - run synchronous training in executor to avoid blocking async loop
        logger.info(f"Starting RL training for {total_timesteps} timesteps")
        
        try:
            # Use thread executor to run synchronous SB3 training
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                await loop.run_in_executor(
                    None,
                    lambda: agent.train(
                        total_timesteps=total_timesteps,
                        callback=callback
                    ),
                )
            # Flush progress updates from callback queue
            if hasattr(callback, 'flush_progress_updates'):
                await callback.flush_progress_updates()
            logger.info(f"Training completed successfully for {experiment_id}")
            
        except Exception as e:
            logger.error(f"Training loop failed: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            await self.training_channel.publish_error(
                experiment_id,
                type(e).__name__,
                str(e),
                traceback.format_exc()
            )
            await self.event_publisher.publish_error(
                experiment_id,
                type(e).__name__,
                str(e),
                traceback.format_exc()
            )
            raise
    
    async def _finalize_success(self, experiment_id: str, config: Dict[str, Any], 
                               env, ml_run) -> None:
        """Finalize successful training."""
        total_timesteps = config.get("timesteps", 10000)
        
        # Get final metrics
        final_portfolio_value = getattr(env, 'current_portfolio_value', 10000.0)
        final_total_return = (final_portfolio_value - 10000.0) / 10000.0
        final_position_count = getattr(env, 'position_count', 0)
        final_trade_count = getattr(env, 'trade_count', 0)
        
        # Finalize ML tracking
        if self.ml_tracker and ml_run:
            try:
                final_metrics = {
                    "final_portfolio_value": float(final_portfolio_value),
                    "final_total_return": float(final_total_return),
                    "total_positions": int(final_position_count),
                    "total_trades": int(final_trade_count),
                    "training_timesteps": total_timesteps
                }
                await self.ml_tracker.finalize_run(ml_run.id, final_metrics, "completed")
                logger.info(f"Finalized ML tracking run: {ml_run.id}")
            except Exception as e:
                logger.warning(f"Failed to finalize ML tracking run: {e}")
        
        # Ensure experiment progress reflects completion before status update
        await self.training_channel.publish_progress(
            experiment_id,
            current_step=total_timesteps,
            total_steps=total_timesteps,
        )
        
        # Mark as completed
        await self.training_channel.publish_status(
            experiment_id, TrainingStatus.COMPLETED,
            f"Training completed: {total_timesteps} timesteps, final portfolio value: ${final_portfolio_value:,.2f}"
        )
        
        logger.info(f"Training completed for experiment {experiment_id}")
    
    async def _handle_training_error(self, experiment_id: str, error: Exception, ml_run) -> None:
        """Handle training errors."""
        logger.error(f"Training failed for experiment {experiment_id}: {error}")
        
        # Finalize ML tracking as failed
        if self.ml_tracker and ml_run:
            try:
                await self.ml_tracker.finalize_run(ml_run.id, status="failed")
            except Exception as e:
                logger.warning(f"Failed to mark ML tracking run as failed: {e}")
        
        # Mark as failed
        await self.training_channel.publish_status(
            experiment_id, TrainingStatus.FAILED, f"Training failed: {str(error)}"
        )
        
        await self.event_publisher.publish_error(
            experiment_id, type(error).__name__, str(error), traceback.format_exc()
        ) 