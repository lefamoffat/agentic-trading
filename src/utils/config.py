from __future__ import annotations

"""Unified application configuration loader.

This module provides a single source of truth (``app_config``) that merges
all YAML files located in the ``configs/`` directory and applies optional
environment-variable overrides.

* API keys and other secrets continue to live in environment variables / ``.env``.
* Trading-specific runtime parameters (position size, risk limits, etc.) now
  live in ``configs/trading_config.yaml`` instead of ``.env``.
* Users can still override individual trading parameters by exporting an env
  var that follows the pattern ``TRADING__<PARAM_NAME>``, e.g.::

      export TRADING__POSITION_SIZE=20000

The public interface is the singleton ``app_config`` at module import time:

>>> from utils.config import app_config
>>> app_config.trading.position_size
10000.0
"""

from pathlib import Path
from typing import Any, Dict
import os

from pydantic import BaseModel, Field

from src.utils.config_loader import ConfigLoader

class TradingConfig(BaseModel):
    """Schema for values inside ``trading_config.yaml``."""

    position_size: float = Field(..., gt=0, description="Default position size in notional currency units")
    max_drawdown: float = Field(..., ge=0, le=1, description="Maximum portfolio drawdown represented as a fraction (0-1)")
    stop_loss: float = Field(..., ge=0, le=1, description="Stop-loss threshold represented as a fraction (0-1)")

class AppConfig(BaseModel):
    """Top-level application configuration."""

    environment: str = Field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    trading: TradingConfig
    agent: Dict[str, Any] = Field(default_factory=dict, description="Full contents of agent_config.yaml")
    data: Dict[str, Any] = Field(default_factory=dict, description="Full contents of data_config.yaml")
    qlib: Dict[str, Any] = Field(default_factory=dict, description="Full contents of qlib_config.yaml")

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    @classmethod
    def load(cls, config_dir: str | Path = "configs") -> "AppConfig":
        """Load YAML files, apply env overrides, and return a validated ``AppConfig`` instance."""

        loader = ConfigLoader(str(config_dir))

        # --- YAML contents -------------------------------------------------
        raw_trading: Dict[str, Any] = loader.get_trading_config() or {}

        # Flatten commonly used primitives so they are easy to access from code
        trading_dict: Dict[str, Any] = {
            "position_size": raw_trading.get("position", {}).get("size"),
            "max_drawdown": raw_trading.get("risk", {}).get("max_drawdown"),
            "stop_loss": raw_trading.get("risk", {}).get("stop_loss"),
        }

        # Verify we actually obtained the three required keys
        missing_keys = [k for k, v in trading_dict.items() if v is None]
        if missing_keys:
            raise ValueError(
                "Missing required keys in trading_config.yaml: " + ", ".join(missing_keys)
            )

        agent_dict: Dict[str, Any] = loader.get_agent_config() or {}

        # --- Environment-variable overrides -------------------------------
        # e.g. ``TRADING__POSITION_SIZE`` overrides ``position_size``
        for key, value in list(trading_dict.items()):
            env_key = f"TRADING__{key.upper()}"
            if env_key in os.environ:
                trading_dict[key] = type(value)(os.environ[env_key])

        trading = TradingConfig(**trading_dict)

        return cls(
            trading=trading,
            agent=agent_dict,
            data={},  # Empty dict - no data config needed
            qlib={},  # Empty dict - no qlib config needed
        )

# ---------------------------------------------------------------------------
# Public singleton – imported by other modules
# ---------------------------------------------------------------------------

app_config: AppConfig = AppConfig.load() 