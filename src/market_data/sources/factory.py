"""Source factory for creating market data sources."""
from src.types import DataSource, BrokerType
from src.market_data.sources.base import MarketDataSource
from src.market_data.sources.broker_source import BrokerSource
from src.market_data.exceptions import DataSourceError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MarketDataSourceFactory:
    """Creates market data sources using enum dispatch."""
    
    def create_source(self, source: DataSource) -> MarketDataSource:
        """
        Create data source instance.
        
        Credentials and configuration handled internally by each source.
        
        Args:
            source: DataSource enum value
            
        Returns:
            Configured data source instance
            
        Raises:
            DataSourceError: If data source is not implemented
        """
        
        if source == DataSource.FOREX_COM:
            return BrokerSource(BrokerType.FOREX_COM)
        
        else:
            raise DataSourceError(
                f"Data source {source.value} is not implemented. "
                f"Available sources: {self.get_available_sources()}"
            )
    
    def get_available_sources(self) -> list[DataSource]:
        """Get list of actually implemented data sources."""
        return [DataSource.FOREX_COM]
    
    def is_source_available(self, source: DataSource) -> bool:
        """Check if a data source is implemented and available."""
        return source in self.get_available_sources()


# Global factory instance
source_factory = MarketDataSourceFactory() 