import asyncio
import signal
import sys
from src.training import get_training_service
from src.utils.logger import get_logger

logger = get_logger("run_training_service")

async def main():
    logger.info("[SERVICE] Starting persistent TrainingService daemon...")
    training_service = await get_training_service()
    await training_service.start()
    logger.info("[SERVICE] TrainingService started and listening for jobs.")

    # Wait forever, handle graceful shutdown
    stop_event = asyncio.Event()

    def _signal_handler(*_):
        logger.info("[SERVICE] Received shutdown signal. Stopping TrainingService...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Signal handlers not supported (e.g., on Windows)
            pass

    await stop_event.wait()
    logger.info("[SERVICE] Shutting down TrainingService...")
    await training_service.stop()
    logger.info("[SERVICE] TrainingService stopped. Exiting.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"[SERVICE] Fatal error: {e}")
        sys.exit(1) 