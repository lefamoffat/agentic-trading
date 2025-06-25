import asyncio
import redis.asyncio as redis
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def test_redis():
    try:
        logger.info("Connecting to Redis...")
        r = redis.Redis()
        
        logger.info("Testing PING...")
        result = await r.ping()
        logger.info(f"PING result: {result}")
        
        logger.info("Testing SET...")
        await r.set("test_key", "test_value")
        
        logger.info("Testing GET...")
        value = await r.get("test_key")
        logger.info(f"GET result: {value}")
        
        logger.info("Testing DELETE...")
        await r.delete("test_key")
        
        logger.info("All Redis operations successful!")
        
    except Exception as e:
        logger.error(f"Redis test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_redis()) 