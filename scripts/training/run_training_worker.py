import sys
import json
import asyncio
from src.training.service import TrainingService
from src.utils.logger import get_logger
from src.messaging import TrainingStatus
import traceback

logger = get_logger("run_training_worker")

def print_and_log(msg):
    print(msg, flush=True)
    logger.info(msg)

async def main():
    if len(sys.argv) != 3:
        print_and_log("Usage: run_training_worker.py <experiment_id> <config_json>")
        sys.exit(1)
    experiment_id = sys.argv[1]
    config_json = sys.argv[2]
    try:
        config = json.loads(config_json)
    except Exception as e:
        print_and_log(f"Failed to parse config: {e}")
        sys.exit(1)
    print_and_log(f"[WORKER] Starting training for {experiment_id}")
    loop = asyncio.get_running_loop()
    service = TrainingService(loop=loop)
    try:
        print_and_log("[WORKER] Starting TrainingService...")
        await service.start()
        print_and_log("[WORKER] TrainingService started.")
        # Explicitly create the experiment in Redis and publish initial status
        print_and_log("[WORKER] Creating experiment in Redis...")
        await service.training_channel.create_experiment(experiment_id, config)
        await service.event_publisher.publish_status_update(
            experiment_id,
            TrainingStatus.STARTING,
            "Experiment starting... (worker)"
        )
        print_and_log("[WORKER] Experiment created. Starting training execution...")
        # Patch the training executor to print progress and catch all exceptions
        orig_run_training_loop = service.training_executor._run_training_loop
        async def patched_run_training_loop(*args, **kwargs):
            print_and_log("[WORKER] Entered training loop.")
            try:
                await orig_run_training_loop(*args, **kwargs)
            except Exception as e:
                print_and_log(f"[WORKER] Exception in training loop: {e}\n{traceback.format_exc()}")
                raise
        service.training_executor._run_training_loop = patched_run_training_loop
        # Patch execute_training to not call _handle_error
        orig_execute_training = service.training_executor.execute_training
        async def patched_execute_training(*args, **kwargs):
            try:
                await orig_execute_training(*args, **kwargs)
            except Exception as e:
                print_and_log(f"[WORKER] Exception in execute_training: {e}\n{traceback.format_exc()}")
                raise
        service.training_executor.execute_training = patched_execute_training
        await service.training_executor.execute_training(experiment_id, config)
        print_and_log(f"[WORKER] Training completed for {experiment_id}")
    except Exception as e:
        print_and_log(f"[WORKER] Training failed for {experiment_id}: {e}\n{traceback.format_exc()}")
        sys.exit(1)
    finally:
        print_and_log("[WORKER] Stopping TrainingService...")
        await service.stop()
        print_and_log("[WORKER] TrainingService stopped.")

if __name__ == "__main__":
    asyncio.run(main()) 