"""Ensure the Redis broker class loads without optional dependency errors."""
from src.messaging.brokers import get_broker_cls


def test_get_broker_cls_redis():
    """Calling get_broker_cls('redis') should return RedisBroker without error."""
    broker_cls = get_broker_cls("redis")
    assert broker_cls.__name__ == "RedisBroker" 