import os

import uvicorn
from src.messaging.config import get_messaging_config

# Refuse to launch API with the in-memory broker in non-test environments.
cfg = get_messaging_config()
if cfg.broker_type == "memory" and os.getenv("ENV", "dev") != "test":
    raise SystemExit(
        "❌  In-memory broker is for unit tests only.\n"
        "    Export MESSAGE_BROKER_TYPE=redis and ensure Redis is running."
    )

from apps.api import create_app

if __name__ == "__main__":
    uvicorn.run("apps.api:create_app", host="0.0.0.0", port=8000, factory=True, reload=False) 