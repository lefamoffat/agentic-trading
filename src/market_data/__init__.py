"""Market data module for centralized data handling.

This module provides a unified interface for fetching, caching, and preparing
market data from various sources (brokers, APIs, etc.) for training and analysis.

Key Features:
- Source-agnostic data fetching (brokers, yfinance, etc.)
- Intelligent caching to avoid re-downloading
- Qlib binary format integration
- Type-safe data contracts using Pydantic
- Date range based data retrieval (inspired by Alpaca API)

Example Usage:
    # Prepare training data
    df = await prepare_training_data(
        symbol="EUR/USD",
        source=DataSource.FOREX_COM,
        timeframe=Timeframe.H1,
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2023, 12, 31)
    )
    
    # Direct source access
    source = source_factory.create_source(DataSource.FOREX_COM)
    df = await source.get_historical_data(...)
"""

from datetime import datetime
from typing import Optional
import pandas as pd

from src.types import DataSource, Timeframe
from src.market_data.sources.factory import source_factory
from src.market_data.storage.manager import storage_manager
from src.market_data.contracts import MarketDataRequest, MarketDataResponse
from src.market_data.exceptions import DataSourceError, DataRangeError, StorageError
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def prepare_training_data(
    symbol: str,
    source: DataSource,
    timeframe: Timeframe,
    start_date: datetime,
    end_date: datetime,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Prepare training data for a specific symbol and date range.
    
    This is the main entry point for training scripts, replacing subprocess
    orchestration with proper module imports.
    
    Args:
        symbol: Trading symbol (e.g., "EUR/USD")
        source: Data source to use
        timeframe: Data timeframe
        start_date: Start date (UTC)
        end_date: End date (UTC)
        force_refresh: Skip cache and fetch fresh data
        
    Returns:
        DataFrame with OHLCV data, ready for training
        
    Raises:
        DataSourceError: If data fetching fails
        DataRangeError: If date range is invalid
        StorageError: If caching operations fail
    """
    try:
        # Create data request
        request = MarketDataRequest(
            symbol=symbol,
            source=source,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info(f"Preparing training data: {symbol} from {source.value} "
                   f"({timeframe.value}, {start_date.date()} to {end_date.date()})")
        
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_data = await storage_manager.get_cached_data(request)
            if cached_data is not None:
                logger.info(f"Using cached data for {symbol} ({len(cached_data)} bars)")
                return cached_data
        
        # Fetch fresh data from source
        data_source = source_factory.create_source(source)
        
        # Validate symbol support
        if not data_source.supports_symbol(symbol):
            raise DataSourceError(f"Symbol {symbol} not supported by {source.value}")
        
        # Fetch historical data
        df = await data_source.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            raise DataSourceError(f"No data returned for {symbol}")
        
        # Create response object for validation
        response = MarketDataResponse(
            request=request,
            data=df,
            bars_count=len(df),
            actual_start_date=df['timestamp'].min().to_pydatetime(),
            actual_end_date=df['timestamp'].max().to_pydatetime()
        )
        
        logger.info(f"Fetched {response.bars_count} bars "
                   f"({response.actual_start_date.date()} to {response.actual_end_date.date()})")
        
        # Cache the data for future use
        await storage_manager.cache_data(response)
        
        return df
        
    except Exception as e:
        if isinstance(e, (DataSourceError, DataRangeError, StorageError)):
            raise
        else:
            logger.error(f"Unexpected error preparing training data: {e}")
            raise DataSourceError(f"Failed to prepare training data: {e}") from e

async def get_available_symbols(source: DataSource) -> list[str]:
    """
    Get list of symbols supported by a data source.
    
    Args:
        source: Data source to query
        
    Returns:
        List of supported symbols
        
    Raises:
        DataSourceError: If source is not available
    """
    try:
        data_source = source_factory.create_source(source)
        
        # For now, we'll use the broker's symbol mapper
        # This could be extended to query live symbol lists
        if source == DataSource.FOREX_COM:
            from src.brokers.symbol_mapper import SymbolMapper, BrokerType
            symbol_mapper = SymbolMapper(BrokerType.FOREX_COM)
            return symbol_mapper.get_supported_symbols()
        
        return []
        
    except Exception as e:
        logger.error(f"Failed to get symbols for {source.value}: {e}")
        raise DataSourceError(f"Cannot get symbols for {source.value}: {e}") from e

def get_available_sources() -> list[DataSource]:
    """Get list of available data sources."""
    return source_factory.get_available_sources()

def is_source_available(source: DataSource) -> bool:
    """Check if a data source is available."""
    return source_factory.is_source_available(source)

# Import components from the consolidated data modules
from src.market_data.calendars import BaseCalendar, ForexCalendar, CalendarFactory, calendar_factory
from src.market_data.processing import DataProcessor
from src.market_data.download import prepare_for_qlib, download_historical_data, download_and_save_qlib_data
from src.market_data.features import build_features

# Export key components for direct access if needed
__all__ = [
    # Main functions
    "prepare_training_data",
    "get_available_symbols", 
    "get_available_sources",
    "is_source_available",
    
    # Factories for advanced usage
    "source_factory",
    "storage_manager",
    
    # Data contracts
    "MarketDataRequest",
    "MarketDataResponse",
    
    # Exceptions
    "DataSourceError",
    "DataRangeError", 
    "StorageError",
    
    # Calendars (migrated from src.data)
    "BaseCalendar",
    "ForexCalendar",
    "CalendarFactory",
    "calendar_factory",
    
    # Processing (migrated from src.data)
    "DataProcessor",
    
    # Download functionality (migrated from scripts)
    "prepare_for_qlib",
    "download_historical_data", 
    "download_and_save_qlib_data",
    
    # Features functionality (migrated from scripts)
    "build_features"
] 