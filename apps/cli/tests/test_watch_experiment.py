"""Regression test for CLI live watch error handling.

If the WebSocket handshake fails (e.g. API returns HTTP 500), the
``_watch_experiment`` helper should handle the exception internally and
exit gracefully without propagating errors to the caller.  Previously this
crashed the CLI with an ``InvalidStatus`` traceback.

This test patches the ``ws_connect`` helper to raise
``websockets.exceptions.InvalidStatus`` immediately, simulating the broken
handshake observed in manual testing.  The function is expected to catch
the exception, print a helpful message, and return normally.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest
from websockets.exceptions import InvalidStatus

from apps.cli.commands import training


@pytest.mark.asyncio
async def test_watch_experiment_handles_ws_failure(monkeypatch, capsys):
    """``_watch_experiment`` should not crash when WS connect fails."""

    @asynccontextmanager
    async def failing_ws_connect(_path):  # noqa: D401 (simple)
        # Simulate server responding with HTTP 500 during handshake
        raise InvalidStatus("mock HTTP 500")
        yield  # pragma: no cover

    # Replace real ws_connect with failing stub
    monkeypatch.setattr(training, "ws_connect", failing_ws_connect)

    # Should return without raising – previously raised InvalidStatus
    await training._watch_experiment("dummy_id", interval=0)

    captured = capsys.readouterr()
    # Check that the graceful error message reached stdout
    assert "Real-time updates unavailable" in captured.out 