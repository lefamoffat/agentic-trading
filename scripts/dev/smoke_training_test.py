#!/usr/bin/env python3
"""Quick manual smoke-test for the end-to-end training pipeline.

Run with

    uv run scripts/dev/smoke_training_test.py

Prerequisites:
    1. A Redis instance listening on localhost:6379  ( `podman run --rm -p 6379:6379 docker.io/library/redis:7-alpine` )
    2. Aim server optional but not required – the test simply checks that a run ID is created.

The script starts the `TrainingService`, launches a 20-step experiment and
polls Redis until it is marked *completed* or until a timeout is reached.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from typing import Any, Dict

from src.training import get_training_service
from src.messaging import TrainingStatus
from src.utils.logger import get_logger

logger = get_logger("smoke_test")

DEFAULT_TIMEOUT = 60  # seconds

CONFIG: Dict[str, Any] = {
    "symbol": "EUR/USD",
    "timeframe": "1h",
    "timesteps": 20,
    "agent_type": "ppo",
}


async def main() -> None:
    experiment_id = f"smoke_{uuid.uuid4().hex[:8]}"
    service = await get_training_service()
    await service.start()

    logger.info(f"🚀 Launching smoke test experiment {experiment_id}")
    await service.start_experiment(experiment_id, CONFIG)

    # Poll Redis via channel API
    start_time = time.time()
    last_step = 0
    while time.time() - start_time < DEFAULT_TIMEOUT:
        exp = await service.training_channel.get_experiment(experiment_id)
        if exp:
            curr = int(exp.get("current_step", 0))
            total = int(exp.get("total_steps", 0))
            if curr != last_step:
                logger.info(f"Progress {curr}/{total}")
                last_step = curr
            if exp.get("status") == TrainingStatus.COMPLETED.value:
                logger.info("✅ Experiment completed successfully – smoke test PASS")
                break
        await asyncio.sleep(1)
    else:
        logger.error("❌ Smoke test timed-out – progress did not complete in time")
        sys.exit(1)

    await service.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted")
        sys.exit(1) 