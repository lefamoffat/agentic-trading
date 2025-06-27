"""Tests for the broker-type guard in ``apps.api.main``.

These checks guarantee that the API refuses to start when the in-memory
message broker is configured outside the test environment, as enforced by
``apps/api/main.py``.
"""

from __future__ import annotations

import importlib
import sys

import pytest


@pytest.mark.parametrize("env_value", [None, "dev", "prod", "staging"])
def test_api_main_refuses_memory(monkeypatch, env_value):
    """Importing ``apps.api.main`` with MEMORY broker must exit."""
    monkeypatch.setenv("MESSAGE_BROKER_TYPE", "memory")
    if env_value is not None:
        monkeypatch.setenv("ENV", env_value)
    else:
        monkeypatch.delenv("ENV", raising=False)

    # Reload the module under the modified env
    sys.modules.pop("apps.api.main", None)
    with pytest.raises(SystemExit) as excinfo:
        importlib.import_module("apps.api.main")  # noqa: WPS433 (dynamic import for test)

    # Ensure our custom error message is present
    assert "In-memory broker is for unit tests only" in str(excinfo.value)


def test_api_main_allows_redis(monkeypatch):
    """Import succeeds when REDIS broker is configured."""
    monkeypatch.setenv("MESSAGE_BROKER_TYPE", "redis")
    monkeypatch.setenv("ENV", "dev")

    sys.modules.pop("apps.api.main", None)
    # Should not raise
    importlib.import_module("apps.api.main")  # noqa: WPS433 