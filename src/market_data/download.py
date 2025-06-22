"""Historical data download functionality.

This module downloads historical data from various sources
and stores it in the appropriate format for the trading system.
"""

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.brokers.factory import broker_factory
from src.market_data.processing import DataProcessor
from src.market_data.storage.manager import storage_manager
from src.market_data.contracts import MarketDataRequest, MarketDataResponse
from src.types import BrokerType, Timeframe, DataSource
from src.utils.logger import get_logger
from src.utils.settings import Settings

logger = get_logger(__name__)


def prepare_for_qlib(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a DataFrame for Qlib's data format.

    - Renames columns to lowercase ('open', 'high', 'low', 'close', 'volume').
    - Renames the timestamp column to 'date'.
    - Adds a 'factor' column with a constant value of 1.0.
    - Sets 'date' as the index.

    Args:
        df: The standardized DataFrame to process.

    Returns:
        A DataFrame formatted for Qlib.
    """
    df = df.copy()
    df.rename(
        columns={
            "timestamp": "date",
            # The following are already lowercase from the data processor
            # "Open": "open",
            # "High": "high", 
            # "Low": "low",
            # "Close": "close",
            # "Volume": "volume",
        },
        inplace=True,
    )

    # The 'date' column must first be converted to a proper datetime type.
    df["date"] = pd.to_datetime(df["date"])

    # Now that it's a datetime object, we can safely remove the timezone.
    # Qlib's data loader script expects timezone-naive timestamps.
    df["date"] = df["date"].dt.tz_localize(None)

    df["factor"] = 1.0
    df.set_index("date", inplace=True)
    return df


async def download_historical_data(
    bars: int = 365,
    symbol: str = "EUR/USD",
    timeframe: str = Timeframe.H1.value,
    broker: str = BrokerType.FOREX_COM.value
) -> Optional[pd.DataFrame]:
    """Download historical data from specified broker with intelligent caching.

    Args:
        bars: Number of bars to download
        symbol: Symbol to download data for (default: EUR/USD)
        timeframe: Timeframe for data (5m, 15m, 1h, 4h, 1d)
        broker: Broker to download from (default: forex.com)

    Returns:
        DataFrame with standardized historical data, or None if failed
    """
    # Calculate date range (timezone-aware for MarketDataRequest)
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=bars)
    
    # Create market data request for cache checking
    request = MarketDataRequest(
        symbol=symbol,
        source=DataSource.FOREX_COM if broker == BrokerType.FOREX_COM.value else DataSource.FOREX_COM,
        timeframe=Timeframe.from_standard(timeframe),
        start_date=start_date,
        end_date=end_date,
        bars_requested=bars
    )
    
    # Check cache first
    logger.info(f"Checking cache for {symbol} {timeframe} data...")
    cached_data = await storage_manager.get_cached_data(request)
    if cached_data is not None and not cached_data.empty:
        logger.info(f"✅ Found cached data: {len(cached_data)} bars")
        return cached_data
    
    logger.info("💾 Cache miss - downloading fresh data...")
    settings = Settings()

    # Validate broker-specific credentials
    if broker == BrokerType.FOREX_COM.value:
        if not settings.forex_com_username or not settings.forex_com_password:
            logger.error("Missing forex.com credentials. Please set FOREX_COM_USERNAME and FOREX_COM_PASSWORD in .env")
            return None

        if not settings.forex_com_app_key:
            logger.error("Missing FOREX_COM_APP_KEY. Please set it in .env")
            return None

        # Create broker instance using factory
        broker_instance = broker_factory.create_broker(
            broker_name=broker,
            api_key=settings.forex_com_username,
            api_secret=settings.forex_com_password,
            sandbox=settings.forex_com_sandbox
        )
    else:
        logger.error(f"Broker '{broker}' credentials validation not implemented yet")
        return None

    # Initialize data processor
    data_processor = DataProcessor(symbol=symbol, asset_class="forex")

    try:
        # Authenticate
        logger.info(f"Authenticating with {broker}...")
        if not await broker_instance.authenticate():
            logger.error(f"Failed to authenticate with {broker}")
            return None

        logger.info(f"Downloading {symbol} data from {start_date.date()} to {end_date.date()}")
        logger.info(f"Timeframe: {timeframe}, Broker: {broker}")

        # Download data
        raw_df = await broker_instance.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            bars=bars
        )

        if raw_df.empty:
            logger.warning("No data received")
            return None

        # Standardize data format
        logger.info("Standardizing data format...")
        standardized_df = data_processor.standardize_dataframe(raw_df, broker_name=broker)

        if standardized_df.empty:
            logger.warning("No data after standardization")
            return None

        # Validate data quality
        quality_report = data_processor.validate_data_quality(standardized_df)
        logger.info(f"Data quality score: {quality_report['quality_score']:.2f}")
        if quality_report['issues']:
            logger.warning(f"Data quality issues: {quality_report['issues']}")

        # Create response object and cache the data
        response = MarketDataResponse(
            request=request,
            data=standardized_df,
            actual_start_date=standardized_df['timestamp'].min(),
            actual_end_date=standardized_df['timestamp'].max(),
            bars_count=len(standardized_df)
        )
        
        # Cache the data for future use
        logger.info("💾 Caching downloaded data...")
        await storage_manager.cache_data(response)

        return standardized_df

    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return None


async def download_and_save_qlib_data(
    bars: int = 365,
    symbol: str = "EUR/USD", 
    timeframe: str = Timeframe.H1.value,
    broker: str = BrokerType.FOREX_COM.value
) -> Optional[Path]:
    """Download historical data and save it in Qlib format.
    
    Args:
        bars: Number of bars to download
        symbol: Symbol to download data for
        timeframe: Timeframe for data
        broker: Broker to download from
        
    Returns:
        Path to saved Qlib CSV file, or None if failed
    """
    # Download standardized data
    standardized_df = await download_historical_data(bars, symbol, timeframe, broker)
    if standardized_df is None:
        return None
    
    # Prepare for Qlib
    logger.info("Preparing data for Qlib...")
    qlib_df = prepare_for_qlib(standardized_df)

    # Create data directory for Qlib source files
    sanitized_symbol = symbol.replace("/", "")
    qlib_source_dir = Path(f"data/qlib_source/{timeframe}")
    qlib_source_dir.mkdir(parents=True, exist_ok=True)

    # Save data in Qlib-compatible CSV format
    filepath = qlib_source_dir / f"{sanitized_symbol}.csv"
    qlib_df.to_csv(filepath)
    logger.info(f"Successfully saved Qlib-ready data to {filepath}")
    
    return filepath


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Download historical data")
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
        default=Timeframe.D1.value,
        choices=Timeframe.get_display_list(),
        help="Timeframe for data (default: 1d)"
    )
    parser.add_argument(
        "--broker",
        type=str,
        default=BrokerType.FOREX_COM.value,
        choices=[BrokerType.FOREX_COM.value],
        help="Broker to download data from (default: forex.com)"
    )

    args = parser.parse_args()

    print("🚀 Historical Data Downloader")
    print("=" * 40)
    print(f"Broker: {args.broker}")
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Bars: {args.bars}")
    print("=" * 40)

    if args.broker == BrokerType.FOREX_COM.value:
        asyncio.run(download_and_save_qlib_data(args.bars, args.symbol, args.timeframe, args.broker))
    else:
        logger.error(f"Broker '{args.broker}' not yet implemented")


if __name__ == "__main__":
    main() 