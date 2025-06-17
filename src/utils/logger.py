"""Logging configuration using loguru
"""

import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from src.utils.settings import settings


def setup_logging(log_level: Optional[str] = None):
    """Set up logging configuration.
    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    # Get log level from environment or default to INFO
    if log_level is None:
        log_level = settings.log_level.upper()

    log_config = {
        "handlers": [
            {
                "sink": "sys.stdout",
                "format": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
                "level": log_level,
            }
        ]
    }

    # Add file handler if logs_path is configured
    try:
        log_file = settings.logs_path / "agent.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_config["handlers"].append(
            {
                "sink": log_file,
                "level": log_level,
                "format": (
                    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                    "{name}:{function}:{line} | {message}"
                ),
                "rotation": "10 MB",
                "retention": "30 days",
                "enqueue": True,  # Make logging non-blocking
                "backtrace": True,
                "diagnose": True,
            }
        )
    except Exception as e:
        logger.warning(f"Could not configure file logging: {e}")

    logger.configure(**log_config)


def get_logger(name: Optional[str] = None):
    """Get a logger instance.
    Args:
        name: Logger name (usually __name__).
    Returns:
        Logger instance.
    """
    # Setup logging if not already configured
    if not logger._core.handlers:
        setup_logging()

    return logger.bind(name=name) if name else logger


# Setup default logging
setup_logging()
