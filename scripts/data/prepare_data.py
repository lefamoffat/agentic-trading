#!/usr/bin/env python3
"""
A master script to prepare all data needed for training.

This script orchestrates the entire data pipeline:
1. Downloads historical data.
2. Dumps the data into Qlib's binary format.
3. Builds features from the binary data.
"""
import subprocess
import argparse
from pathlib import Path
import shutil

from src.utils.logger import get_logger
from src.types import Timeframe

def run_command(command: list[str], description: str):
    """Run a command and handle errors."""
    logger = get_logger(__name__)
    logger.info(f"--- Running: {description} ---")
    try:
        subprocess.run(command, check=True)
        logger.info(f"--- Finished: {description} ---")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run '{' '.join(command)}': {e}")
        raise
    except FileNotFoundError:
        logger.error(
            f"Could not find the script for '{description}'. Make sure you are in the project root."
        )
        raise

def main():
    """Main function to run the data preparation pipeline."""
    parser = argparse.ArgumentParser(description="Prepare all data for training.")
    parser.add_argument(
        "--symbol",
        type=str,
        default="EUR/USD",
        help="The trading symbol to use (e.g., EUR/USD)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1d",
        help="The timeframe for the data (e.g., 1d)",
    )
    args = parser.parse_args()
    
    sanitized_symbol = args.symbol.replace("/", "")
    
    print("🚀 Starting Full Data Preparation Pipeline")
    print("=" * 40)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print("=" * 40)

    # --- Step 0: Clean and Prepare Directories ---
    qlib_data_path = Path("data/qlib_data")
    if qlib_data_path.exists():
        print(f"🧹 Cleaning existing Qlib data directory: {qlib_data_path}")
        shutil.rmtree(qlib_data_path)
    print(f"✨ Creating new Qlib data directory: {qlib_data_path}")
    qlib_data_path.mkdir(parents=True)

    # --- Step 1: Download Historical Data ---
    download_command = [
        "python", "-m", "scripts.data.download_historical",
        "--symbol", args.symbol,
        "--timeframe", args.timeframe,
    ]
    run_command(download_command, "Download Historical Data")

    # --- Step 2: Dump Data to Qlib Binary Format ---
    qlib_source_path = f"data/qlib_source/{args.timeframe}"
    
    # Get the correct frequency name for Qlib
    qlib_freq = Timeframe.from_standard(args.timeframe).qlib_name

    dump_command = [
        "python", "-m", "scripts.data.dump_bin", "dump_all",
        "--csv_path", qlib_source_path,
        "--qlib_dir", str(qlib_data_path),
        "--freq", qlib_freq,
        "--date_field_name", "date",  # The CSVs are prepared with a 'date' column
        "--symbol_field_name", "symbol",
    ]
    run_command(dump_command, "Dump to Qlib Binary")
    
    # --- Step 3: Build Features ---
    build_command = [
        "python", "-m", "scripts.features.build_features",
        "--symbol", args.symbol,
        "--timeframe", args.timeframe,
    ]
    run_command(build_command, "Build Features")
    
    print("=" * 40)
    print("✅ Data preparation complete.")
    print("=" * 40)

if __name__ == "__main__":
    main() 