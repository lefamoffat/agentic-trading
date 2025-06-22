"""Tests for main market_data interface."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch
import pandas as pd
import pytest

from src.market_data import (
    prepare_training_data,
    get_available_symbols,
    get_available_sources,
    is_source_available
)
from src.market_data.exceptions import DataSourceError, DataRangeError, StorageError
from src.types import DataSource, Timeframe

@pytest.mark.unit
class TestMainInterface:
    """Test main market_data interface functions."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample OHLCV DataFrame."""
        return pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1, i, tzinfo=timezone.utc) for i in range(24)],
            'open': [1.1000 + i * 0.0001 for i in range(24)],
            'high': [1.1010 + i * 0.0001 for i in range(24)],
            'low': [1.0990 + i * 0.0001 for i in range(24)],
            'close': [1.1005 + i * 0.0001 for i in range(24)],
            'volume': [1000 + i * 10 for i in range(24)]
        })
    
    @patch('src.market_data.storage_manager')
    @patch('src.market_data.source_factory')
    @pytest.mark.asyncio
    async def test_prepare_training_data_cache_hit(self, mock_source_factory, mock_storage_manager, sample_dataframe):
        """Test prepare_training_data with cache hit."""
        # Setup mocks
        mock_storage_manager.get_cached_data = AsyncMock(return_value=sample_dataframe)
        
        # Call function
        result = await prepare_training_data(
            symbol="EUR/USD",
            source=DataSource.FOREX_COM,
            timeframe=Timeframe.H1,
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc)
        )
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 24
        mock_storage_manager.get_cached_data.assert_called_once()
        mock_source_factory.create_source.assert_not_called()  # Should not fetch from source
    
    @patch('src.market_data.storage_manager')
    @patch('src.market_data.source_factory')
    @pytest.mark.asyncio
    async def test_prepare_training_data_cache_miss(self, mock_source_factory, mock_storage_manager, sample_dataframe):
        """Test prepare_training_data with cache miss."""
        # Setup mocks
        mock_storage_manager.get_cached_data = AsyncMock(return_value=None)  # Cache miss
        mock_storage_manager.cache_data = AsyncMock()
        
        mock_source = Mock()
        mock_source.supports_symbol.return_value = True
        mock_source.get_historical_data = AsyncMock(return_value=sample_dataframe)
        mock_source_factory.create_source.return_value = mock_source
        
        # Call function
        result = await prepare_training_data(
            symbol="EUR/USD",
            source=DataSource.FOREX_COM,
            timeframe=Timeframe.H1,
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc)
        )
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 24
        mock_storage_manager.get_cached_data.assert_called_once()
        mock_source_factory.create_source.assert_called_once_with(DataSource.FOREX_COM)
        mock_source.supports_symbol.assert_called_once_with("EUR/USD")
        mock_source.get_historical_data.assert_called_once()
        mock_storage_manager.cache_data.assert_called_once()
    
    @patch('src.market_data.storage_manager')
    @patch('src.market_data.source_factory')
    @pytest.mark.asyncio
    async def test_prepare_training_data_force_refresh(self, mock_source_factory, mock_storage_manager, sample_dataframe):
        """Test prepare_training_data with force_refresh=True."""
        # Setup mocks
        mock_storage_manager.cache_data = AsyncMock()
        
        mock_source = Mock()
        mock_source.supports_symbol.return_value = True
        mock_source.get_historical_data = AsyncMock(return_value=sample_dataframe)
        mock_source_factory.create_source.return_value = mock_source
        
        # Call function with force_refresh
        result = await prepare_training_data(
            symbol="EUR/USD",
            source=DataSource.FOREX_COM,
            timeframe=Timeframe.H1,
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
            force_refresh=True
        )
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        mock_storage_manager.get_cached_data.assert_not_called()  # Should skip cache check
        mock_source.get_historical_data.assert_called_once()
    
    @patch('src.market_data.source_factory')
    @pytest.mark.asyncio
    async def test_prepare_training_data_unsupported_symbol(self, mock_source_factory):
        """Test prepare_training_data with unsupported symbol."""
        # Setup mocks
        mock_source = Mock()
        mock_source.supports_symbol.return_value = False  # Symbol not supported
        mock_source_factory.create_source.return_value = mock_source
        
        # Call function and expect error
        with pytest.raises(DataSourceError, match="Symbol .* not supported"):
            await prepare_training_data(
                symbol="UNSUPPORTED",
                source=DataSource.FOREX_COM,
                timeframe=Timeframe.H1,
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 2, tzinfo=timezone.utc)
            )
    
    @patch('src.market_data.storage_manager')
    @patch('src.market_data.source_factory')
    @pytest.mark.asyncio
    async def test_prepare_training_data_empty_response(self, mock_source_factory, mock_storage_manager):
        """Test prepare_training_data with empty data response."""
        # Setup mocks
        mock_storage_manager.get_cached_data = AsyncMock(return_value=None)
        
        mock_source = Mock()
        mock_source.supports_symbol.return_value = True
        mock_source.get_historical_data = AsyncMock(return_value=pd.DataFrame())  # Empty DataFrame
        mock_source_factory.create_source.return_value = mock_source
        
        # Call function and expect error
        with pytest.raises(DataSourceError, match="No data returned"):
            await prepare_training_data(
                symbol="EUR/USD",
                source=DataSource.FOREX_COM,
                timeframe=Timeframe.H1,
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 2, tzinfo=timezone.utc)
            )
    
    @patch('src.market_data.storage_manager')
    @patch('src.market_data.source_factory')
    @pytest.mark.asyncio
    async def test_prepare_training_data_source_error(self, mock_source_factory, mock_storage_manager):
        """Test prepare_training_data with source error."""
        # Setup mocks
        mock_storage_manager.get_cached_data = AsyncMock(return_value=None)
        
        mock_source = Mock()
        mock_source.supports_symbol.return_value = True
        mock_source.get_historical_data = AsyncMock(side_effect=Exception("Source error"))
        mock_source_factory.create_source.return_value = mock_source
        
        # Call function and expect error
        with pytest.raises(DataSourceError, match="Failed to prepare training data"):
            await prepare_training_data(
                symbol="EUR/USD",
                source=DataSource.FOREX_COM,
                timeframe=Timeframe.H1,
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 2, tzinfo=timezone.utc)
            )
    
    @patch('src.market_data.source_factory')
    @pytest.mark.asyncio
    async def test_get_available_symbols_forex_com(self, mock_source_factory):
        """Test get_available_symbols for forex.com."""
        # Setup mocks
        mock_source = Mock()
        mock_source_factory.create_source.return_value = mock_source
        
        with patch('src.brokers.symbol_mapper.SymbolMapper') as mock_mapper_class:
            mock_mapper = Mock()
            mock_mapper.get_supported_symbols.return_value = ["EUR/USD", "GBP/USD", "USD/JPY"]
            mock_mapper_class.return_value = mock_mapper
            
            # Call function
            symbols = await get_available_symbols(DataSource.FOREX_COM)
            
            # Assertions
            assert isinstance(symbols, list)
            assert "EUR/USD" in symbols
            assert "GBP/USD" in symbols
            assert "USD/JPY" in symbols
            assert len(symbols) == 3
    
    @patch('src.market_data.source_factory')
    @pytest.mark.asyncio
    async def test_get_available_symbols_error(self, mock_source_factory):
        """Test get_available_symbols with error."""
        mock_source_factory.create_source.side_effect = Exception("Source error")
        
        with pytest.raises(DataSourceError, match="Cannot get symbols"):
            await get_available_symbols(DataSource.FOREX_COM)
    
    @patch('src.market_data.source_factory')
    def test_get_available_sources(self, mock_source_factory):
        """Test get_available_sources function."""
        mock_source_factory.get_available_sources.return_value = [DataSource.FOREX_COM]
        
        sources = get_available_sources()
        
        assert isinstance(sources, list)
        assert DataSource.FOREX_COM in sources
        mock_source_factory.get_available_sources.assert_called_once()
    
    @patch('src.market_data.source_factory')
    def test_is_source_available(self, mock_source_factory):
        """Test is_source_available function."""
        mock_source_factory.is_source_available.return_value = True
        
        result = is_source_available(DataSource.FOREX_COM)
        
        assert result is True
        mock_source_factory.is_source_available.assert_called_once_with(DataSource.FOREX_COM)

@pytest.mark.integration
class TestMarketDataIntegration:
    """Integration tests for market_data module (require real components)."""
    
    def test_module_imports(self):
        """Test that all main module components can be imported."""
        from src.market_data import (
            prepare_training_data,
            get_available_symbols,
            get_available_sources,
            is_source_available,
            source_factory,
            storage_manager,
            MarketDataRequest,
            MarketDataResponse,
            DataSourceError,
            DataRangeError,
            StorageError
        )
        
        # Basic smoke test - just check they exist
        assert callable(prepare_training_data)
        assert callable(get_available_symbols)
        assert callable(get_available_sources)
        assert callable(is_source_available)
        assert source_factory is not None
        assert storage_manager is not None
        assert MarketDataRequest is not None
        assert MarketDataResponse is not None
        assert issubclass(DataSourceError, Exception)
        assert issubclass(DataRangeError, Exception)
        assert issubclass(StorageError, Exception)
    
    def test_available_sources_real(self):
        """Test getting available sources with real factory."""
        sources = get_available_sources()
        
        assert isinstance(sources, list)
        assert len(sources) > 0
        assert DataSource.FOREX_COM in sources
    
    def test_source_availability_real(self):
        """Test source availability with real factory."""
        assert is_source_available(DataSource.FOREX_COM)
    
    def test_create_real_request(self):
        """Test creating real MarketDataRequest."""
        from src.market_data.contracts import MarketDataRequest
        
        request = MarketDataRequest(
            symbol="EUR/USD",
            source=DataSource.FOREX_COM,
            timeframe=Timeframe.H1,
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc)
        )
        
        assert request.symbol == "EUR/USD"
        assert request.source == DataSource.FOREX_COM
        assert request.timeframe == Timeframe.H1
        
        # Test cache key generation
        cache_key = request.get_cache_key()
        assert isinstance(cache_key, str)
        assert len(cache_key) > 0 