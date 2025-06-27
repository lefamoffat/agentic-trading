from __future__ import annotations

"""Tiny async client for talking to the Agentic Trading API.

All CLI commands import this module to perform HTTP and WebSocket operations
against the REST service. The base URL is discovered via the ``PUBLIC_API_URL``
environment variable.

If the variable is missing or the API is unreachable, commands will exit with
an actionable error.
"""

import os
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any, Dict

import httpx
import websockets
from websockets.exceptions import ConnectionClosed


API_URL = os.getenv("PUBLIC_API_URL")
if not API_URL:
    raise RuntimeError(
        "Environment variable PUBLIC_API_URL is not set. Start the API "
        "service and export PUBLIC_API_URL=http://127.0.0.1:8000 (or your host).",
    )

# Ensure no trailing slash for clean path joins
API_URL = API_URL.rstrip("/")

# Shared async client – created lazily
_client: httpx.AsyncClient | None = None


async def get_client() -> httpx.AsyncClient:  # noqa: WPS231
    """Return a singleton AsyncClient with sane defaults."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(base_url=API_URL, timeout=30.0, follow_redirects=True)

        # Verify the API is using a production-ready broker
        health_resp = await _client.get("/health")
        health_resp.raise_for_status()
        backend = health_resp.json().get("messaging_backend", {})
        if backend.get("broker_type") == "memory":
            raise RuntimeError(
                "iAPI is running with the in-memory message broker, which is NOT supported!"
            )
    return _client


async def get(path: str, **kwargs: Any) -> httpx.Response:  # noqa: WPS110
    client = await get_client()
    return await client.get(path, **kwargs)


async def post(path: str, **kwargs: Any) -> httpx.Response:
    client = await get_client()
    return await client.post(path, **kwargs)


@asynccontextmanager
async def ws(path: str) -> AsyncGenerator[websockets.WebSocketClientProtocol, None]:
    """Async context manager to connect to a WebSocket relative to base URL."""
    # Convert http(s) base to ws scheme
    if API_URL.startswith("https://"):
        ws_base = "wss://" + API_URL[len("https://") :]
    elif API_URL.startswith("http://"):
        ws_base = "ws://" + API_URL[len("http://") :]
    else:
        # Assume already ws://
        ws_base = API_URL
    url = f"{ws_base}{path}"
    connection = await websockets.connect(url, ping_interval=20, ping_timeout=20)
    try:
        yield connection
    finally:
        await connection.close()


async def health_check() -> Dict[str, Any]:
    """Quick check to see if API is reachable (raises on failure)."""
    resp = await get("/")
    resp.raise_for_status()
    return resp.json() 