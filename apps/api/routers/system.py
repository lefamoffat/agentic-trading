from __future__ import annotations

"""System-level endpoints (health, broker info)."""
from fastapi import APIRouter

from src.messaging.config import get_messaging_config

router = APIRouter()


@router.get("/health", summary="Basic service health")
async def health() -> dict:  # noqa: D401 (simple)
    cfg = get_messaging_config()
    return {
        "status": "ok",
        "messaging_backend": {
            "broker_type": cfg.broker_type,
            "production_ready": cfg.broker_type != "memory",
        },
    } 