"""Broker-based market data source implementation."""
from datetime import datetime, timezone
from typing import List
import pandas as pd

from src.types import BrokerType, Timeframe
from src.brokers.factory import broker_factory
from src.market_data.sources.base import MarketDataSource
from src.market_data.exceptions import DataSourceError, DataRangeError
from src.utils.logger import get_logger

logger = get_logger(__name__)

class BrokerSource(MarketDataSource):
    """Market data source that wraps existing broker functionality."""
    
    def __init__(self, broker: BrokerType):
        """
        Initialize broker source.
        
        Args:
            broker: Type of broker to use (e.g., BrokerType.FOREX_COM)
        """
        self.broker = broker
        self._broker_instance = None
        logger.info(f"Initialized BrokerSource for {broker.value}")
    
    async def _get_broker(self):
        """Get or create authenticated broker instance. Credentials handled by broker factory."""
        if self._broker_instance is None:
            try:
                # Use broker factory method that handles credential loading internally
                self._broker_instance = broker_factory.create_broker_with_env_credentials(
                    broker_type=self.broker,
                    sandbox=True
                )
                
                # Authenticate the broker
                authenticated = await self._broker_instance.authenticate()
                if not authenticated:
                    raise DataSourceError(f"Broker authentication failed for {self.broker.value}")
                
                logger.info(f"Successfully connected to {self.broker.value} broker")
                
            except ValueError as e:
                # Credential or broker type errors
                logger.error(f"Broker configuration error: {e}")
                raise DataSourceError(f"Broker configuration failed: {e}") from e
            except Exception as e:
                logger.error(f"Failed to initialize {self.broker.value} broker: {e}")
                raise DataSourceError(f"Broker initialization failed: {e}") from e
        
        return self._broker_instance
    
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data using the existing broker.
        
        Args:
            symbol: Trading symbol (e.g., "EUR/USD")
            timeframe: Data timeframe enum
            start_date: Start date (UTC)
            end_date: End date (UTC)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            
        Raises:
            DataSourceError: If broker operations fail
            DataRangeError: If date range calculation fails
        """
        try:
            broker_instance = await self._get_broker()
            
            # Calculate number of bars from date range and timeframe
            bars = self._calculate_bars_from_date_range(start_date, end_date, timeframe)
            
            if bars <= 0:
                raise DataRangeError(f"Invalid date range: {start_date} to {end_date}")
            
            logger.info(f"Fetching {bars} bars of {symbol} {timeframe.value} data from broker")
            
            # Use existing broker get_historical_data method
            # Note: This gets the LATEST bars, not historical range
            # This is a limitation of the current broker API that takes bars instead of date range
            df = await broker_instance.get_historical_data(
                symbol=symbol,
                timeframe=timeframe.value,
                bars=bars
            )
            
            if df.empty:
                raise DataSourceError(f"No data returned from broker for {symbol}")
            
            # Ensure timestamps are UTC
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            
            # Sort by timestamp (oldest first)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Retrieved {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            if isinstance(e, (DataSourceError, DataRangeError)):
                raise
            else:
                logger.error(f"Broker data retrieval failed: {e}")
                raise DataSourceError(f"Failed to get historical data from broker: {e}") from e
    
    def supports_symbol(self, symbol: str) -> bool:
        """
        Check if this broker supports the given symbol.
        
        Args:
            symbol: Trading symbol to check
            
        Returns:
            True if symbol is supported, False otherwise
        """
        try:
            if self.broker == BrokerType.FOREX_COM:
                # Use the existing symbol mapper to check support
                from src.brokers.symbol_mapper import SymbolMapper
                symbol_mapper = SymbolMapper(BrokerType.FOREX_COM)
                supported_symbols = symbol_mapper.get_supported_symbols()
                return symbol in supported_symbols
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking symbol support for {symbol}: {e}")
            return False
    
    def _calculate_bars_from_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        timeframe: Timeframe
    ) -> int:
        """
        Calculate approximate number of bars for the given date range.
        
        Note: This is approximate since it doesn't account for weekends/holidays.
        The broker API limitation means we can't get exact date ranges.
        """
        try:
            # Validate date range
            if start_date >= end_date:
                raise DataRangeError(f"Start date {start_date} must be before end date {end_date}")
            
            # Calculate time difference
            time_diff = end_date - start_date
            total_minutes = time_diff.total_seconds() / 60
            
            # Get timeframe in minutes
            timeframe_minutes = timeframe.minutes
            
            # Calculate approximate bars
            bars = int(total_minutes / timeframe_minutes)
            
            # Add some buffer for weekends/holidays
            bars = int(bars * 1.4)  # 40% buffer
            
            # Ensure reasonable limits
            bars = max(1, min(bars, 5000))  # Between 1 and 5000 bars
            
            logger.debug(f"Calculated {bars} bars for {start_date} to {end_date} at {timeframe.value}")
            
            return bars
            
        except DataRangeError:
            # Re-raise DataRangeError as-is
            raise
        except Exception as e:
            logger.error(f"Error calculating bars from date range: {e}")
            raise DataRangeError(f"Failed to calculate bars from date range: {e}") from e 