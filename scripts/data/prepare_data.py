#!/usr/bin/env python3
"""A master script to prepare all data needed for training.

This script orchestrates the entire data pipeline using the new market_data module:
1. Uses market_data.prepare_training_data() for centralized data handling
2. Eliminates subprocess orchestration in favor of proper module imports
3. Provides a simple CLI interface for the new centralized approach
"""
import argparse
import asyncio
from datetime import datetime, timedelta, timezone

from src.types import Timeframe, DataSource
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def main():
    """Main function to run the data preparation pipeline."""
    parser = argparse.ArgumentParser(description="Prepare training data using market_data module.")
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
        help="The timeframe for the data (e.g., 1d, 1h)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of historical data to fetch (default: 365)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="forex.com",
        choices=["forex.com"],
        help="Data source to use (default: forex.com)",
    )
    args = parser.parse_args()

    print("🚀 Starting Data Preparation with market_data module")
    print("=" * 50)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Days: {args.days}")
    print(f"Source: {args.source}")
    print("=" * 50)

    try:
        # Convert string arguments to enums
        timeframe = Timeframe.from_standard(args.timeframe)
        source = DataSource.FOREX_COM  # Only supported source currently
        
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=args.days)
        
        logger.info(f"Fetching {args.symbol} data from {start_date.date()} to {end_date.date()}")
        
        # Use the new centralized market_data module
        from src.market_data import prepare_training_data
        
        df = await prepare_training_data(
            symbol=args.symbol,
            source=source,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            force_refresh=False  # Use cache if available
        )
        
        logger.info(f"✅ Successfully prepared {len(df)} bars of training data")
        logger.info(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"Columns: {list(df.columns)}")
        
        print("=" * 50)
        print("✅ Data preparation complete using market_data module.")
        print("🚀 Ready for training!")
        print("=" * 50)
        
    except Exception as e:
        logger.error(f"❌ Data preparation failed: {e}")
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
