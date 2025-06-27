import asyncio
from typing import Dict, Any, List

from src.messaging.config import get_messaging_config
from src.messaging.brokers.redis import RedisBroker
from src.utils.logger import get_logger


logger = get_logger("purge_invalid_experiments")


async def _purge(broker: RedisBroker) -> None:  # noqa: D401 (simple)
    """Scan Redis and delete malformed experiment hashes.

    Criteria for deletion:
    1. ``timesteps``, ``learning_rate`` or ``initial_balance`` are missing or
       not strictly greater than zero.
    2. ``current_step`` numerically exceeds ``total_steps``.
    """
    purged: List[str] = []

    # Load all experiment IDs registered in the active set
    redis = broker._redis  # type: ignore[attr-defined] (intentionally accessing)
    if redis is None:
        await broker.connect()
        redis = broker._redis  # type: ignore[attr-defined]
        assert redis is not None

    exp_ids = await redis.smembers("active_experiments")
    logger.info("Scanning %d experiments for corruption", len(exp_ids))

    for exp_id in exp_ids:
        key = f"experiment:{exp_id}"
        data: Dict[str, Any] = await redis.hgetall(key)
        if not data:
            await redis.srem("active_experiments", exp_id)
            purged.append(exp_id)
            continue

        try:
            timesteps = int(data.get("timesteps", "0"))
            learning_rate = float(data.get("learning_rate", "0"))
            initial_balance = float(data.get("initial_balance", "0"))
            current_step = int(data.get("current_step", "0"))
            total_steps = int(data.get("total_steps", "0"))
        except ValueError:
            # Non-numeric where numeric expected → purge
            await redis.delete(key)
            await redis.srem("active_experiments", exp_id)
            purged.append(exp_id)
            continue

        invalid_config = timesteps <= 0 or learning_rate <= 0 or initial_balance <= 0
        invalid_progress = total_steps > 0 and current_step > total_steps

        if invalid_config or invalid_progress:
            await redis.delete(key)
            await redis.srem("active_experiments", exp_id)
            purged.append(exp_id)
            logger.info("Purged %s (invalid_config=%s, invalid_progress=%s)", exp_id, invalid_config, invalid_progress)

    logger.info("Purged %d corrupt experiment records", len(purged))


async def main() -> None:  # noqa: D401 (simple)
    cfg = get_messaging_config().redis
    broker = RedisBroker(cfg)
    await _purge(broker)
    await broker.close()


if __name__ == "__main__":
    asyncio.run(main()) 