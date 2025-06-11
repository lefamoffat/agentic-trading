"""
Logging configuration using loguru
"""

import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(
    log_level: str = None,
    log_file: Optional[str] = None,
    enable_console: bool = True
):
    """
    Setup logging configuration
    
    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        enable_console: Whether to enable console logging
    """
    # Get log level from environment or default to INFO
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")
    
    # Remove default logger
    logger.remove()
    
    # Console logging
    if enable_console:
        logger.add(
            sys.stdout,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True
        )
    
    # File logging
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )


def get_logger(name: str = None):
    """
    Get a logger instance
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    # Setup logging if not already configured
    if not logger._core.handlers:
        setup_logging()
    
    if name:
        return logger.bind(name=name)
    return logger


# Setup default logging
setup_logging(
    log_file="logs/system/agentic_trading.log"
) 