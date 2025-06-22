"""Abstract base interface for market data sources."""
from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd

from src.types import Timeframe

class MarketDataSource(ABC):
    """Abstract base class for all market data sources."""
    
    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data for the specified date range.
        
        Args:
            symbol: Trading symbol (e.g., "EUR/USD")
            timeframe: Data timeframe enum
            start_date: Start date (UTC)
            end_date: End date (UTC)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            
        Raises:
            DataSourceError: If data source operations fail
            DataRangeError: If requested date range is invalid
        """
        pass
    
    @abstractmethod
    def supports_symbol(self, symbol: str) -> bool:
        """
        Check if this source supports the given symbol.
        
        Args:
            symbol: Trading symbol to check
            
        Returns:
            True if symbol is supported, False otherwise
        """
        pass 