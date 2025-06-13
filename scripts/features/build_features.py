#!/usr/bin/env python3
"""
Build features using Qlib.

This script initializes Qlib, loads raw data, calculates a set of
alpha factors, and saves the resulting feature set to a CSV file.

Usage:
    python -m scripts.features.build_features --symbol EUR/USD --timeframe 1h
"""
import argparse
from pathlib import Path

import pandas as pd
import qlib
from qlib.data import D

# A selection of factors from Alpha360. We can expand this list.
# See: https://qlib.readthedocs.io/en/latest/component/meta_dataset.html#alpha-360
ALPHA_FACTORS = [
    "Rank(Stddev(Correlation(AvgPrice(5), Log(Volume(20)))), 15))",
    "Rank(Correlation(Sum(Log(Volume(5))), Sum(AvgPrice(60))))",
    "Rank(Max(Correlation(Rank(Volume), Rank(AvgPrice)), 5))",
    "Rank(Sum(Correlation(Rank(Volume), Rank(AvgPrice), 8)))",
    "Rank(Max(Abs(Correlation(AvgPrice, Log(Volume), 15))))",
    "Rank(Avg(Correlation(Rank(AvgPrice), Rank(Log(Volume)), 3)))",
    "Rank(Correlation(AvgPrice, Log(Volume), 10))",
    "Rank(Max(Correlation(Rank(AvgPrice), Rank(Log(Volume)), 15)))",
]


def _get_qlib_freq(timeframe: str) -> str:
    """Converts a timeframe string to a Qlib-compatible frequency string."""
    if timeframe.endswith("h"):
        return f"{int(timeframe[:-1]) * 60}min"
    if timeframe.endswith("d"):
        return timeframe
    # You can add more conversions here if needed (e.g., for weeks 'W' or months 'M')
    # For now, we'll raise an error for unsupported formats.
    raise ValueError(f"Unsupported timeframe for Qlib: {timeframe}")


def build_features(symbol: str, timeframe: str) -> None:
    """
    Build features for a given symbol and timeframe using Qlib.

    Args:
        symbol (str): The trading symbol (e.g., "EUR/USD").
        timeframe (str): The data timeframe (e.g., "1h").
    """
    sanitized_symbol = symbol.replace("/", "")
    
    # Define paths
    raw_data_dir = Path(f"data/raw/historical/{sanitized_symbol}/{timeframe}")
    qlib_data_dir = Path(f"data/qlib_format/qlib_data")
    output_dir = Path("data/processed/features")
    
    # Create directories if they don't exist
    qlib_data_dir.mkdir(parents=True, exist_ok=True)
    (qlib_data_dir / "calendars").mkdir(exist_ok=True)
    (qlib_data_dir / "instruments").mkdir(exist_ok=True)
    (qlib_data_dir / "features").mkdir(exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Prepare data for Qlib ---
    all_files = list(raw_data_dir.glob("*.csv"))
    if not all_files:
        print(f"No raw data found for {symbol} at {raw_data_dir}")
        return

    df_list = [pd.read_csv(f) for f in all_files]
    raw_df = pd.concat(df_list, ignore_index=True)
    raw_df.sort_values(by="timestamp", inplace=True)
    
    raw_df.rename(
        columns={"timestamp": "datetime", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"},
        inplace=True,
    )
    raw_df["instrument"] = sanitized_symbol
    
    qlib_csv_path = qlib_data_dir / f"{sanitized_symbol.lower()}.csv"
    raw_df.to_csv(qlib_csv_path, index=False)

    # --- Step 2: Initialize Qlib ---
    print("Initializing Qlib...")
    qlib.init(provider_uri=str(qlib_data_dir.resolve()))
    print("Qlib initialized successfully.")

    # --- Step 3: Define fields and fetch features from Qlib ---
    print("Defining fields and fetching features from Qlib...")
    
    # Prefixing with '$' tells Qlib to use the raw value. Formulas are used directly.
    fields = ["$open", "$high", "$low", "$close", "$volume"] + ALPHA_FACTORS

    # Convert the timeframe to a Qlib-compatible frequency
    qlib_freq = _get_qlib_freq(timeframe)
    
    # Fetch all data, including calculated alpha factors
    all_df = D.features(
        instruments=[sanitized_symbol],
        fields=fields,
        start_time=raw_df["datetime"].min(),
        end_time=raw_df["datetime"].max(),
        freq=qlib_freq,
    )

    # --- Step 4: Clean up and Save ---
    all_df = all_df.reset_index()
    all_df.rename(columns={"datetime": "timestamp", "instrument": "symbol"}, inplace=True)

    # Clean up column names for compatibility
    new_cols = {}
    for col in all_df.columns:
        if '$' in col:
            new_cols[col] = col.replace('$', '')
        else:
            new_cols[col] = col.replace('(', '_').replace(')', '').replace(',', '').replace(' ', '')
    all_df.rename(columns=new_cols, inplace=True)
    
    final_df = all_df.drop(columns=["symbol"])
    
    output_path = output_dir / f"{sanitized_symbol}_{timeframe}_features.csv"
    final_df.to_csv(output_path, index=False)
    
    print(f"✅ Features built successfully!")
    print(f"Saved {len(final_df.columns)} features for {len(final_df)} timesteps to {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Build features using Qlib")
    parser.add_argument(
        "--symbol",
        type=str,
        default="EUR/USD",
        help="The trading symbol to process (e.g., EUR/USD)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="The timeframe of the data (e.g., 1h)",
    )
    
    args = parser.parse_args()

    print("🚀 Starting Feature Building Process")
    print("=" * 40)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print("=" * 40)
    
    build_features(symbol=args.symbol, timeframe=args.timeframe)


if __name__ == "__main__":
    main() 