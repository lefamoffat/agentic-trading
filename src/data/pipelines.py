#!/usr/bin/env python3
"""
Orchestrates the data preparation pipelines.
"""
import subprocess
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_data_preparation_pipeline(symbol: str, timeframe: str) -> bool:
    """
    Execute the full data preparation pipeline.

    This involves running the following scripts in order:
    1. download_historical_data
    2. dump_to_qlib
    3. build_features

    This is orchestrated by the `prepare_data.py` script.

    Args:
        symbol (str): The trading symbol (e.g., 'EUR/USD').
        timeframe (str): The timeframe of the data (e.g., '1d').

    Returns:
        bool: True if the pipeline ran successfully, False otherwise.
    """
    logger.info("Running data preparation script...")
    try:
        subprocess.run(
            [
                "python",
                "-m",
                "scripts.data.prepare_data",
                "--symbol",
                symbol,
                "--timeframe",
                timeframe,
            ],
            check=True,
            capture_output=True, # Capture stdout/stderr
            text=True, # Decode stdout/stderr as text
        )
        logger.info("Data preparation pipeline completed successfully.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Data preparation pipeline failed: {e}")
        if isinstance(e, subprocess.CalledProcessError):
            logger.error(f"Stderr: {e.stderr}")
        return False 