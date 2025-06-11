#!/usr/bin/env python3
"""
Download historical data for EUR/USD trading.

This script downloads historical EUR/USD data from various sources
and stores it in the appropriate format for the trading system.

Usage:
    python -m scripts.data.download_historical [options]
"""

import asyncio
import argparse
from datetime import datetime, timedelta
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.settings import Settings
from src.brokers.forex_com import ForexComBroker


async def download_forex_data(bars: int = 365, symbol: str = "EUR/USD", timeframe: str = "1h") -> None:
    """
    Download historical forex data.
    
    Args:
        bars: Number of bars to download
        timeframe: Timeframe for data (5m, 15m, 1h, 4h, 1d)
        symbol: Symbol to download data for (default: EUR/USD)
    """
    logger = get_logger(__name__)
    settings = Settings()
    
    # Check if we have forex.com credentials
    if not settings.forex_com_username or not settings.forex_com_password:
        logger.error("Missing forex.com credentials. Please set FOREX_COM_USERNAME and FOREX_COM_PASSWORD in .env")
        return
    
    if not settings.forex_com_app_key:
        logger.error("Missing FOREX_COM_APP_KEY. Please set it in .env")
        return
    
    # Initialize broker
    broker = ForexComBroker(
        api_key=settings.forex_com_username,
        api_secret=settings.forex_com_password,
        sandbox=settings.forex_com_sandbox
    )
    
    try:
        # Authenticate
        logger.info("Authenticating with forex.com...")
        if not await broker.authenticate():
            logger.error("Failed to authenticate with forex.com")
            return
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=bars)
        
        logger.info(f"Downloading {symbol} data from {start_date.date()} to {end_date.date()}")
        logger.info(f"Timeframe: {timeframe}")
        
        # Download data
        df = await broker.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            bars=bars
        )
        
        if df.empty:
            logger.warning("No data received")
            return
        
        # Create data directory with timeframe structure
        data_dir = Path(f"data/raw/historical/{symbol}/{timeframe}/")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data with simplified filename
        filename = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        filepath = data_dir / filename
        
        df.to_csv(filepath)
        logger.info(f"Saved {len(df)} records to {filepath}")
        
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
    

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download historical EUR/USD data")
    parser.add_argument(
        "--bars", 
        type=int, 
        default=365, 
        help="Number of bars to download (default: 365)"
    )
    parser.add_argument(
        "--symbol", 
        type=str, 
        default="EUR/USD", 
        help="Symbol to download data for (default: EUR/USD)"
    )
    parser.add_argument(
        "--timeframe", 
        type=str, 
        default="1h", 
        choices=["5m", "15m", "1h", "4h", "1d"],
        help="Timeframe for data (default: 1h)"
    )
    
    args = parser.parse_args()
    
    print("🚀 Historical Data Downloader")
    print("=" * 40)
    
    asyncio.run(download_forex_data(args.bars, args.symbol, args.timeframe))


if __name__ == "__main__":
    main() 