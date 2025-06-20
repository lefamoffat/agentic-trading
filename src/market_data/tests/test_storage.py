"""Tests for market data storage components."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import pandas as pd
import pytest

from src.market_data.storage.cache import CacheManager
from src.market_data.storage.qlib_integration import QlibConverter
from src.market_data.storage.manager import StorageManager, storage_manager
from src.market_data.contracts import MarketDataRequest, MarketDataResponse, CacheMetadata
from src.market_data.exceptions import StorageError
from src.types import DataSource, Timeframe


@pytest.mark.unit
class TestCacheManager:
    """Test CacheManager functionality."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Create CacheManager with temporary directory."""
        return CacheManager(temp_cache_dir)
    
    @pytest.fixture
    def sample_request(self):
        """Create sample market data request."""
        return MarketDataRequest(
            symbol="EUR/USD",
            source=DataSource.FOREX_COM,
            timeframe=Timeframe.H1,
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc)
        )
    
    @pytest.fixture
    def sample_response(self, sample_request):
        """Create sample market data response."""
        df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1, i, tzinfo=timezone.utc) for i in range(24)],
            'open': [1.1000 + i * 0.0001 for i in range(24)],
            'high': [1.1010 + i * 0.0001 for i in range(24)],
            'low': [1.0990 + i * 0.0001 for i in range(24)],
            'close': [1.1005 + i * 0.0001 for i in range(24)],
            'volume': [1000 + i * 10 for i in range(24)]
        })
        
        return MarketDataResponse(
            request=sample_request,
            data=df,
            bars_count=24,
            actual_start_date=datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
            actual_end_date=datetime(2024, 1, 1, 23, tzinfo=timezone.utc)
        )
    
    def test_cache_manager_initialization(self, cache_manager, temp_cache_dir):
        """Test CacheManager initialization."""
        assert cache_manager.cache_dir == Path(temp_cache_dir)
        assert cache_manager.cache_dir.exists()
        assert cache_manager.metadata_file.name == "cache_metadata.json"
    
    @pytest.mark.asyncio
    async def test_cache_data_success(self, cache_manager, sample_response):
        """Test successful data caching."""
        await cache_manager.cache_data(sample_response)
        
        # Check that metadata was created
        cache_key = sample_response.request.get_cache_key()
        assert cache_key in cache_manager._metadata_cache
        
        metadata = cache_manager._metadata_cache[cache_key]
        assert metadata.symbol == "EUR/USD"
        assert metadata.bars_count == 24
        assert metadata.file_size_bytes > 0
    
    @pytest.mark.asyncio
    async def test_get_cached_data_hit(self, cache_manager, sample_response):
        """Test cache hit scenario."""
        # First cache the data
        await cache_manager.cache_data(sample_response)
        
        # Create a request that is within the cached data range
        # Request data from 00:00 to 22:00 (which should be covered by cached data 00:00 to 23:00)
        subset_request = MarketDataRequest(
            symbol=sample_response.request.symbol,
            source=sample_response.request.source,
            timeframe=sample_response.request.timeframe,
            start_date=datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 1, 22, tzinfo=timezone.utc)
        )
        
        # Should find cached data that covers this range
        cached_df = await cache_manager.get_cached_data(subset_request)
        
        assert cached_df is not None
        assert isinstance(cached_df, pd.DataFrame)
        assert len(cached_df) == 23  # 23 hours of data (0-22 inclusive)
        assert 'timestamp' in cached_df.columns
        
        # Verify the data is correctly filtered to the requested range
        assert cached_df['timestamp'].min() >= subset_request.start_date
        assert cached_df['timestamp'].max() <= subset_request.end_date
    
    @pytest.mark.asyncio
    async def test_get_cached_data_miss(self, cache_manager, sample_request):
        """Test cache miss scenario."""
        cached_df = await cache_manager.get_cached_data(sample_request)
        assert cached_df is None
    
    @pytest.mark.asyncio
    async def test_get_cached_data_expired(self, cache_manager, sample_response):
        """Test expired cache handling."""
        # Cache the data
        await cache_manager.cache_data(sample_response)
        
        # Manually set cache to expired
        cache_key = sample_response.request.get_cache_key()
        metadata = cache_manager._metadata_cache[cache_key]
        old_cached_at = datetime.now(timezone.utc) - pd.Timedelta(hours=25)  # 25 hours ago
        metadata.cached_at = old_cached_at
        
        # Should return None for expired cache
        cached_df = await cache_manager.get_cached_data(sample_response.request)
        assert cached_df is None
    
    def test_cleanup_expired(self, cache_manager, temp_cache_dir):
        """Test cleanup of expired cache entries."""
        # Create mock expired metadata
        expired_metadata = CacheMetadata(
            request_hash="expired_key",
            symbol="EUR/USD",
            source=DataSource.FOREX_COM,
            timeframe=Timeframe.H1,
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
            bars_count=24,
            cached_at=datetime.now(timezone.utc) - pd.Timedelta(hours=25),  # Expired
            file_path=str(Path(temp_cache_dir) / "expired.parquet"),
            file_size_bytes=1024
        )
        
        # Create the cache file
        cache_file = Path(expired_metadata.file_path)
        cache_file.touch()
        
        # Add to metadata cache
        cache_manager._metadata_cache["expired_key"] = expired_metadata
        
        # Cleanup with 24 hour max age
        cleaned_count = cache_manager.cleanup_expired(max_age_hours=24.0)
        
        assert cleaned_count == 1
        assert "expired_key" not in cache_manager._metadata_cache
        assert not cache_file.exists()
    
    def test_get_cache_stats(self, cache_manager):
        """Test cache statistics generation."""
        stats = cache_manager.get_cache_stats()
        
        assert isinstance(stats, dict)
        assert "total_entries" in stats
        assert "total_size_mb" in stats
        assert "unique_symbols" in stats
        assert "unique_sources" in stats
    
    def test_clear_cache(self, cache_manager, temp_cache_dir):
        """Test clearing all cache data."""
        # Add mock metadata
        cache_manager._metadata_cache["test_key"] = CacheMetadata(
            request_hash="test_key",
            symbol="EUR/USD",
            source=DataSource.FOREX_COM,
            timeframe=Timeframe.H1,
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
            bars_count=24,
            cached_at=datetime.now(timezone.utc),
            file_path=str(Path(temp_cache_dir) / "test.parquet"),
            file_size_bytes=1024
        )
        
        # Create cache file
        cache_file = Path(temp_cache_dir) / "test.parquet"
        cache_file.touch()
        
        # Clear cache
        cleared_count = cache_manager.clear_cache()
        
        assert cleared_count == 1
        assert len(cache_manager._metadata_cache) == 0
        assert not cache_file.exists()


@pytest.mark.unit
class TestQlibConverter:
    """Test QlibConverter functionality."""
    
    @pytest.fixture
    def temp_qlib_dir(self):
        """Create temporary qlib directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def qlib_converter(self, temp_qlib_dir):
        """Create QlibConverter with temporary directory."""
        return QlibConverter(temp_qlib_dir)
    
    @pytest.fixture
    def sample_response(self):
        """Create sample market data response for qlib conversion."""
        request = MarketDataRequest(
            symbol="EUR/USD",
            source=DataSource.FOREX_COM,
            timeframe=Timeframe.H1,
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc)
        )
        
        df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1, i, tzinfo=timezone.utc) for i in range(24)],
            'open': [1.1000 + i * 0.0001 for i in range(24)],
            'high': [1.1010 + i * 0.0001 for i in range(24)],
            'low': [1.0990 + i * 0.0001 for i in range(24)],
            'close': [1.1005 + i * 0.0001 for i in range(24)],
            'volume': [1000 + i * 10 for i in range(24)]
        })
        
        return MarketDataResponse(
            request=request,
            data=df,
            bars_count=24,
            actual_start_date=datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
            actual_end_date=datetime(2024, 1, 1, 23, tzinfo=timezone.utc)
        )
    
    def test_qlib_converter_initialization(self, qlib_converter, temp_qlib_dir):
        """Test QlibConverter initialization."""
        assert qlib_converter.qlib_dir == Path(temp_qlib_dir)
        assert qlib_converter.qlib_dir.exists()
    
    def test_get_qlib_path(self, qlib_converter, sample_response):
        """Test qlib path generation."""
        path = qlib_converter.get_qlib_path(sample_response.request)
        
        assert "EUR_USD" in path  # Symbol with slash replaced
        assert "1h.bin" in path
        assert str(qlib_converter.qlib_dir) in path
    
    @pytest.mark.asyncio
    async def test_convert_data_success(self, qlib_converter, sample_response):
        """Test successful data conversion to qlib format."""
        qlib_path = await qlib_converter.convert_data(sample_response)
        
        assert os.path.exists(qlib_path)
        assert qlib_path.endswith("1h.bin")
        
        # Check file has content
        file_size = os.path.getsize(qlib_path)
        assert file_size > 0
    
    def test_prepare_qlib_data(self, qlib_converter, sample_response):
        """Test DataFrame preparation for qlib format."""
        prepared_df = qlib_converter._prepare_qlib_data(sample_response.data)
        
        assert 'date' in prepared_df.columns
        assert '$open' in prepared_df.columns
        assert '$high' in prepared_df.columns
        assert '$low' in prepared_df.columns
        assert '$close' in prepared_df.columns
        assert '$volume' in prepared_df.columns
        
        # Check date format (YYYYMMDD)
        assert prepared_df['date'].iloc[0] == 20240101
    
    def test_prepare_qlib_data_missing_columns(self, qlib_converter):
        """Test qlib data preparation with missing columns."""
        incomplete_df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            'open': [1.1000],
            # Missing high, low, close, volume
        })
        
        with pytest.raises(StorageError, match="Missing required columns"):
            qlib_converter._prepare_qlib_data(incomplete_df)
    
    @pytest.mark.asyncio
    async def test_read_qlib_binary_roundtrip(self, qlib_converter, sample_response):
        """Test writing and reading qlib binary format."""
        # Convert to qlib format
        qlib_path = await qlib_converter.convert_data(sample_response)
        
        # Read it back
        read_df = qlib_converter.read_qlib_binary(qlib_path)
        
        assert isinstance(read_df, pd.DataFrame)
        assert len(read_df) == 24
        assert 'timestamp' in read_df.columns
        assert 'open' in read_df.columns
        assert 'high' in read_df.columns
        assert 'low' in read_df.columns
        assert 'close' in read_df.columns
        assert 'volume' in read_df.columns
    
    def test_read_qlib_binary_nonexistent_file(self, qlib_converter):
        """Test reading non-existent qlib binary file."""
        with pytest.raises(StorageError, match="Qlib binary file not found"):
            qlib_converter.read_qlib_binary("/nonexistent/path.bin")
    
    def test_get_qlib_info_nonexistent_symbol(self, qlib_converter):
        """Test getting qlib info for non-existent symbol."""
        info = qlib_converter.get_qlib_info("NONEXISTENT")
        
        assert info["exists"] is False
        assert info["symbol"] == "NONEXISTENT"


@pytest.mark.unit
class TestStorageManager:
    """Test StorageManager coordination."""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def storage_manager_instance(self, temp_storage_dir):
        """Create StorageManager with temporary directory."""
        return StorageManager(temp_storage_dir)
    
    @pytest.fixture
    def sample_request(self):
        """Create sample market data request."""
        return MarketDataRequest(
            symbol="EUR/USD",
            source=DataSource.FOREX_COM,
            timeframe=Timeframe.H1,
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc)
        )
    
    @pytest.fixture
    def sample_response(self, sample_request):
        """Create sample market data response."""
        df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1, i, tzinfo=timezone.utc) for i in range(24)],
            'open': [1.1000 + i * 0.0001 for i in range(24)],
            'high': [1.1010 + i * 0.0001 for i in range(24)],
            'low': [1.0990 + i * 0.0001 for i in range(24)],
            'close': [1.1005 + i * 0.0001 for i in range(24)],
            'volume': [1000 + i * 10 for i in range(24)]
        })
        
        return MarketDataResponse(
            request=sample_request,
            data=df,
            bars_count=24,
            actual_start_date=datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
            actual_end_date=datetime(2024, 1, 1, 23, tzinfo=timezone.utc)
        )
    
    def test_storage_manager_initialization(self, storage_manager_instance, temp_storage_dir):
        """Test StorageManager initialization."""
        assert storage_manager_instance.storage_dir == Path(temp_storage_dir)
        assert storage_manager_instance.cache_dir.exists()
        assert storage_manager_instance.qlib_dir.exists()
        assert hasattr(storage_manager_instance, 'cache_manager')
        assert hasattr(storage_manager_instance, 'qlib_converter')
    
    @pytest.mark.asyncio
    async def test_cache_data_delegation(self, storage_manager_instance, sample_response):
        """Test that cache_data delegates to cache manager."""
        with patch.object(storage_manager_instance.cache_manager, 'cache_data') as mock_cache:
            await storage_manager_instance.cache_data(sample_response)
            mock_cache.assert_called_once_with(sample_response)
    
    @pytest.mark.asyncio
    async def test_get_cached_data_delegation(self, storage_manager_instance, sample_request):
        """Test that get_cached_data delegates to cache manager."""
        with patch.object(storage_manager_instance.cache_manager, 'get_cached_data') as mock_get:
            mock_get.return_value = None
            
            result = await storage_manager_instance.get_cached_data(sample_request)
            
            assert result is None
            mock_get.assert_called_once_with(sample_request)
    
    @pytest.mark.asyncio
    async def test_convert_to_qlib_delegation(self, storage_manager_instance, sample_response):
        """Test that convert_to_qlib delegates to qlib converter."""
        with patch.object(storage_manager_instance.qlib_converter, 'convert_data') as mock_convert:
            mock_convert.return_value = "/path/to/qlib.bin"
            
            result = await storage_manager_instance.convert_to_qlib(sample_response)
            
            assert result == "/path/to/qlib.bin"
            mock_convert.assert_called_once_with(sample_response)
    
    def test_get_storage_info(self, storage_manager_instance):
        """Test storage information retrieval."""
        info = storage_manager_instance.get_storage_info()
        
        assert isinstance(info, dict)
        assert "storage_dir" in info
        assert "cache_dir" in info
        assert "qlib_dir" in info
        assert "cache_size_mb" in info
        assert "qlib_size_mb" in info
        assert "total_size_mb" in info
        assert "cache_stats" in info
    
    def test_global_storage_manager_instance(self):
        """Test that global storage_manager exists."""
        assert storage_manager is not None
        assert isinstance(storage_manager, StorageManager) 