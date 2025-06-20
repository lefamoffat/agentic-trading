"""Market data coordinator - main API for data preparation."""
from datetime import datetime
from pathlib import Path

from src.types import Timeframe, DataSource
from src.market_data.contracts import DateRange
from src.market_data.exceptions import DataSourceError
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def prepare_for_training(
    symbol: str,
    timeframe: Timeframe,
    source: DataSource,
    start_date: datetime,
    end_date: datetime
) -> Path:
    """
    Prepare market data for training using exact date range.
    
    Args:
        symbol: Trading symbol (EUR/USD, etc.)
        timeframe: Data timeframe enum
        source: Data source enum (only FOREX_COM currently available)
        start_date: Start date for historical data (UTC)
        end_date: End date for historical data (UTC)
        
    Returns:
        Path to training-ready features CSV
        
    Example:
        # Train on 2008 financial crisis data
        features_path = await prepare_for_training(
            symbol="EUR/USD",
            timeframe=Timeframe.D1,
            source=DataSource.FOREX_COM,
            start_date=datetime(2008, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2009, 1, 1, tzinfo=timezone.utc)
        )
    """
    
    # Validate date range
    date_range = DateRange(start_date=start_date, end_date=end_date)
    
    # Phase 1 implementation - placeholder showing intended API
    raise DataSourceError(
        "prepare_for_training() implementation pending. "
        "This is the API placeholder - full implementation in Phases 2-4. "
        f"Would prepare {symbol} {timeframe.value} data from {start_date} to {end_date}"
    )


def _get_features_path(symbol: str, timeframe: Timeframe, source: DataSource) -> Path:
    """Get path to features file."""
    sanitized_symbol = symbol.replace("/", "")
    features_dir = Path("data/processed/features") / source.value
    features_dir.mkdir(parents=True, exist_ok=True)
    return features_dir / f"{sanitized_symbol}_{timeframe.value}_features.csv" 