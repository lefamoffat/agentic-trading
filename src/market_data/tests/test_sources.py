"""Tests for market data sources."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch
import pandas as pd
import pytest

from src.market_data.sources.factory import MarketDataSourceFactory, source_factory
from src.market_data.sources.broker_source import BrokerSource
from src.market_data.sources.base import MarketDataSource
from src.market_data.exceptions import DataSourceError, DataRangeError
from src.types import DataSource, BrokerType, Timeframe

@pytest.mark.unit
class TestMarketDataSourceFactory:
    """Test MarketDataSourceFactory."""
    
    @pytest.fixture
    def factory(self):
        """Create factory instance for testing."""
        return MarketDataSourceFactory()
    
    def test_factory_initialization(self, factory):
        """Test factory initialization."""
        assert hasattr(factory, 'create_source')
        assert hasattr(factory, 'get_available_sources')
        assert hasattr(factory, 'is_source_available')
    
    def test_create_forex_com_source(self, factory):
        """Test creating forex.com data source."""
        source = factory.create_source(DataSource.FOREX_COM)
        
        assert isinstance(source, BrokerSource)
        assert source.broker == BrokerType.FOREX_COM
    
    def test_create_unknown_source_raises_error(self, factory):
        """Test creating unknown source raises DataSourceError."""
        # Create a mock DataSource that doesn't exist
        with patch('src.types.DataSource') as mock_enum:
            mock_unknown = Mock()
            mock_unknown.value = "unknown_source"
            
            with pytest.raises(DataSourceError, match="Data source unknown_source is not implemented"):
                factory.create_source(mock_unknown)
    
    def test_get_available_sources(self, factory):
        """Test getting list of available sources."""
        sources = factory.get_available_sources()
        
        assert isinstance(sources, list)
        assert DataSource.FOREX_COM in sources
        assert len(sources) >= 1
    
    def test_is_source_available(self, factory):
        """Test checking source availability."""
        assert factory.is_source_available(DataSource.FOREX_COM)
        
        # Test with mock unknown source
        with patch('src.types.DataSource') as mock_enum:
            mock_unknown = Mock()
            mock_unknown.value = "unknown_source"
            assert not factory.is_source_available(mock_unknown)
    
    def test_global_factory_instance(self):
        """Test that global source_factory exists and works."""
        assert source_factory is not None
        assert isinstance(source_factory, MarketDataSourceFactory)
        
        # Test it can create sources
        source = source_factory.create_source(DataSource.FOREX_COM)
        assert isinstance(source, BrokerSource)

@pytest.mark.unit
class TestBrokerSource:
    """Test BrokerSource implementation."""
    
    @pytest.fixture
    def broker_source(self):
        """Create BrokerSource for testing."""
        return BrokerSource(BrokerType.FOREX_COM)
    
    @pytest.fixture
    def mock_broker(self):
        """Create mock broker instance."""
        mock = Mock()
        mock.authenticate = AsyncMock(return_value=True)
        mock.get_historical_data = AsyncMock()
        return mock
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV DataFrame."""
        return pd.DataFrame({
            'timestamp': [
                datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 2, tzinfo=timezone.utc)
            ],
            'open': [1.1000, 1.1010, 1.1020],
            'high': [1.1020, 1.1030, 1.1040],
            'low': [1.0990, 1.1000, 1.1010],
            'close': [1.1010, 1.1020, 1.1030],
            'volume': [1000, 1100, 1200]
        })
    
    def test_broker_source_initialization(self, broker_source):
        """Test BrokerSource initialization."""
        assert broker_source.broker == BrokerType.FOREX_COM
        assert broker_source._broker_instance is None
    
    @patch('src.market_data.sources.broker_source.broker_factory')
    @pytest.mark.asyncio
    async def test_get_broker_success(self, mock_factory, broker_source, mock_broker):
        """Test successful broker creation and authentication."""
        mock_factory.create_broker_with_env_credentials.return_value = mock_broker
        
        broker = await broker_source._get_broker()
        
        assert broker == mock_broker
        mock_factory.create_broker_with_env_credentials.assert_called_once_with(
            broker_type=BrokerType.FOREX_COM,
            sandbox=True
        )
        mock_broker.authenticate.assert_called_once()
    
    @patch('src.market_data.sources.broker_source.broker_factory')
    @pytest.mark.asyncio
    async def test_get_broker_authentication_failure(self, mock_factory, broker_source):
        """Test broker authentication failure."""
        mock_broker = Mock()
        mock_broker.authenticate = AsyncMock(return_value=False)
        mock_factory.create_broker_with_env_credentials.return_value = mock_broker
        
        with pytest.raises(DataSourceError, match="Broker authentication failed"):
            await broker_source._get_broker()
    
    @patch('src.market_data.sources.broker_source.broker_factory')
    @pytest.mark.asyncio
    async def test_get_broker_credential_error(self, mock_factory, broker_source):
        """Test broker credential configuration error."""
        mock_factory.create_broker_with_env_credentials.side_effect = ValueError("Missing credentials")
        
        with pytest.raises(DataSourceError, match="Broker configuration failed"):
            await broker_source._get_broker()
    
    @patch('src.market_data.sources.broker_source.broker_factory')
    @pytest.mark.asyncio
    async def test_get_historical_data_success(self, mock_factory, broker_source, mock_broker, sample_ohlcv_data):
        """Test successful historical data retrieval."""
        mock_factory.create_broker_with_env_credentials.return_value = mock_broker
        mock_broker.get_historical_data.return_value = sample_ohlcv_data
        
        result = await broker_source.get_historical_data(
            symbol="EUR/USD",
            timeframe=Timeframe.H1,
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 1, 3, tzinfo=timezone.utc)
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'timestamp' in result.columns
        mock_broker.get_historical_data.assert_called_once()
    
    @patch('src.market_data.sources.broker_source.broker_factory')
    @pytest.mark.asyncio
    async def test_get_historical_data_empty_response(self, mock_factory, broker_source, mock_broker):
        """Test handling of empty data response."""
        mock_factory.create_broker_with_env_credentials.return_value = mock_broker
        mock_broker.get_historical_data.return_value = pd.DataFrame()  # Empty DataFrame
        
        with pytest.raises(DataSourceError, match="No data returned from broker"):
            await broker_source.get_historical_data(
                symbol="EUR/USD",
                timeframe=Timeframe.H1,
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, 3, tzinfo=timezone.utc)
            )
    
    def test_calculate_bars_from_date_range(self, broker_source):
        """Test bars calculation from date range."""
        start_date = datetime(2024, 1, 1, 0, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 1, 4, tzinfo=timezone.utc)  # 4 hours
        
        bars = broker_source._calculate_bars_from_date_range(
            start_date, end_date, Timeframe.H1
        )
        
        # Should be approximately 4 bars with buffer
        assert bars >= 4
        assert bars <= 10  # With 40% buffer
    
    def test_calculate_bars_invalid_range(self, broker_source):
        """Test bars calculation with invalid date range."""
        start_date = datetime(2024, 1, 2, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 1, tzinfo=timezone.utc)  # Before start
        
        with pytest.raises(DataRangeError):
            broker_source._calculate_bars_from_date_range(
                start_date, end_date, Timeframe.H1
            )
    
    @patch('src.brokers.symbol_mapper.SymbolMapper')
    def test_supports_symbol_forex_com(self, mock_mapper_class, broker_source):
        """Test symbol support checking for forex.com."""
        mock_mapper = Mock()
        mock_mapper.get_supported_symbols.return_value = ["EUR/USD", "GBP/USD"]
        mock_mapper_class.return_value = mock_mapper
        
        assert broker_source.supports_symbol("EUR/USD")
        assert not broker_source.supports_symbol("AAPL")
    
    def test_supports_symbol_error_handling(self, broker_source):
        """Test symbol support checking with error."""
        with patch('src.brokers.symbol_mapper.SymbolMapper', side_effect=Exception("Mapper error")):
            # Should return False on error, not raise
            assert not broker_source.supports_symbol("EUR/USD")

@pytest.mark.unit
class TestMarketDataSourceBase:
    """Test MarketDataSource abstract base class."""
    
    def test_abstract_base_class(self):
        """Test that MarketDataSource cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MarketDataSource()
    
    def test_subclass_must_implement_methods(self):
        """Test that subclasses must implement abstract methods."""
        class IncompleteSource(MarketDataSource):
            pass
        
        with pytest.raises(TypeError):
            IncompleteSource()
    
    def test_complete_subclass_works(self):
        """Test that complete subclass implementation works."""
        class CompleteSource(MarketDataSource):
            async def get_historical_data(self, symbol, timeframe, start_date, end_date):
                return pd.DataFrame()
            
            def supports_symbol(self, symbol):
                return True
        
        # Should not raise
        source = CompleteSource()
        assert isinstance(source, MarketDataSource) 