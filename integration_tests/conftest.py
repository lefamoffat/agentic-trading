import os
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts" / "podman"

REDIS_SCRIPT = SCRIPTS_DIR / "start_redis.sh"
REDIS_STOP = SCRIPTS_DIR / "stop_redis.sh"


@pytest.fixture(scope="session")
def redis_service():
    """Start a disposable Redis via Podman for integration tests."""
    # Skip if running in CI without podman
    if not REDIS_SCRIPT.exists():
        pytest.skip("Podman helper scripts not available")
    try:
        subprocess.run(["bash", str(REDIS_SCRIPT)], check=True)
        # naive wait – rely on health checks inside container
        time.sleep(3)
        yield
    finally:
        subprocess.run(["bash", str(REDIS_STOP)], check=True) 