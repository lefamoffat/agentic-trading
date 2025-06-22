"""Tests for market data contracts (Pydantic models)."""

from datetime import datetime, timezone
import pandas as pd
import pytest
from pydantic import ValidationError

from src.market_data.contracts import (
    MarketDataRequest,
    MarketDataResponse, 
    CacheMetadata,
    QlibDataSpec
)
from src.types import DataSource, Timeframe


@pytest.mark.unit
class TestMarketDataRequest:
    """Test MarketDataRequest Pydantic model."""
    
    def test_valid_request_creation(self):
        """Test creating a valid market data request."""
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
        assert request.start_date.year == 2024
        assert request.end_date.year == 2024
    
    def test_symbol_validation_and_normalization(self):
        """Test symbol validation and uppercase normalization."""
        request = MarketDataRequest(
            symbol="  eur/usd  ",  # lowercase with spaces
            source=DataSource.FOREX_COM,
            timeframe=Timeframe.H1,
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc)
        )
        
        assert request.symbol == "EUR/USD"  # Should be normalized
    
    def test_empty_symbol_validation(self):
        """Test that empty symbol raises validation error."""
        with pytest.raises(ValidationError, match="Symbol cannot be empty"):
            MarketDataRequest(
                symbol="",
                source=DataSource.FOREX_COM,
                timeframe=Timeframe.H1,
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 2, tzinfo=timezone.utc)
            )
    
    def test_date_range_validation(self):
        """Test that end_date must be after start_date."""
        with pytest.raises(ValidationError, match="end_date must be after start_date"):
            MarketDataRequest(
                symbol="EUR/USD",
                source=DataSource.FOREX_COM,
                timeframe=Timeframe.H1,
                start_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc)  # Before start_date
            )
    
    def test_timezone_validation(self):
        """Test that dates must be timezone-aware."""
        with pytest.raises(ValidationError, match="Dates must be timezone-aware"):
            MarketDataRequest(
                symbol="EUR/USD",
                source=DataSource.FOREX_COM,
                timeframe=Timeframe.H1,
                start_date=datetime(2024, 1, 1),  # No timezone
                end_date=datetime(2024, 1, 2, tzinfo=timezone.utc)
            )
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        request = MarketDataRequest(
            symbol="EUR/USD",
            source=DataSource.FOREX_COM,
            timeframe=Timeframe.H1,
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc)
        )
        
        cache_key = request.get_cache_key()
        assert "EUR/USD" in cache_key
        assert "forex_com" in cache_key  # This uses .value from the enum
        assert "1h" in cache_key  # This uses .value from the enum
        assert "2024-01-01" in cache_key
        assert "2024-01-02" in cache_key


@pytest.mark.unit
class TestMarketDataResponse:
    """Test MarketDataResponse Pydantic model."""
    
    @pytest.fixture
    def valid_request(self):
        """Create a valid request for testing."""
        return MarketDataRequest(
            symbol="EUR/USD",
            source=DataSource.FOREX_COM,
            timeframe=Timeframe.H1,
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc)
        )
    
    @pytest.fixture
    def valid_dataframe(self):
        """Create a valid OHLCV DataFrame for testing."""
        return pd.DataFrame({
            'timestamp': [
                datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 1, tzinfo=timezone.utc)
            ],
            'open': [1.1000, 1.1010],
            'high': [1.1020, 1.1030],
            'low': [1.0990, 1.1000],
            'close': [1.1010, 1.1020],
            'volume': [1000, 1100]
        })
    
    def test_valid_response_creation(self, valid_request, valid_dataframe):
        """Test creating a valid market data response."""
        response = MarketDataResponse(
            request=valid_request,
            data=valid_dataframe,
            bars_count=2,
            actual_start_date=datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
            actual_end_date=datetime(2024, 1, 1, 1, tzinfo=timezone.utc)
        )
        
        assert response.request == valid_request
        assert len(response.data) == 2
        assert response.bars_count == 2
        assert not response.cached  # Default value
    
    def test_dataframe_structure_validation(self, valid_request):
        """Test DataFrame structure validation."""
        invalid_df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            'open': [1.1000],
            # Missing required columns: high, low, close, volume
        })
        
        with pytest.raises(ValidationError, match="DataFrame missing required columns"):
            MarketDataResponse(
                request=valid_request,
                data=invalid_df,
                bars_count=1,
                actual_start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                actual_end_date=datetime(2024, 1, 1, tzinfo=timezone.utc)
            )
    
    def test_bars_count_validation(self, valid_request, valid_dataframe):
        """Test that bars_count matches DataFrame length."""
        with pytest.raises(ValidationError, match="bars_count .* doesn't match DataFrame length"):
            MarketDataResponse(
                request=valid_request,
                data=valid_dataframe,
                bars_count=5,  # DataFrame has 2 rows
                actual_start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                actual_end_date=datetime(2024, 1, 1, 1, tzinfo=timezone.utc)
            )
    
    def test_date_range_coverage_calculation(self, valid_request, valid_dataframe):
        """Test date range coverage calculation."""
        response = MarketDataResponse(
            request=valid_request,
            data=valid_dataframe,
            bars_count=2,
            actual_start_date=datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
            actual_end_date=datetime(2024, 1, 1, 12, tzinfo=timezone.utc)  # 12 hours of 24 requested
        )
        
        coverage = response.get_date_range_coverage()
        assert coverage == 0.5  # 12 hours of 24 hours requested
    
    def test_is_complete_coverage(self, valid_request, valid_dataframe):
        """Test complete coverage detection."""
        # Test complete coverage
        response = MarketDataResponse(
            request=valid_request,
            data=valid_dataframe,
            bars_count=2,
            actual_start_date=datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
            actual_end_date=datetime(2024, 1, 2, 0, tzinfo=timezone.utc)
        )
        
        assert response.is_complete_coverage(tolerance_hours=1.0)
        
        # Test incomplete coverage
        response_incomplete = MarketDataResponse(
            request=valid_request,
            data=valid_dataframe,
            bars_count=2,
            actual_start_date=datetime(2024, 1, 1, 6, tzinfo=timezone.utc),  # 6 hours late
            actual_end_date=datetime(2024, 1, 1, 18, tzinfo=timezone.utc)   # 6 hours early
        )
        
        assert not response_incomplete.is_complete_coverage(tolerance_hours=1.0)


@pytest.mark.unit
class TestCacheMetadata:
    """Test CacheMetadata Pydantic model."""
    
    def test_valid_metadata_creation(self):
        """Test creating valid cache metadata."""
        metadata = CacheMetadata(
            request_hash="test_hash",
            symbol="EUR/USD",
            source=DataSource.FOREX_COM,
            timeframe=Timeframe.H1,
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
            bars_count=24,
            cached_at=datetime(2024, 1, 3, tzinfo=timezone.utc),
            file_path="/path/to/cache.parquet",
            file_size_bytes=1024
        )
        
        assert metadata.symbol == "EUR/USD"
        assert metadata.bars_count == 24
        assert metadata.file_size_bytes == 1024
        assert metadata.source == DataSource.FOREX_COM
        assert metadata.timeframe == Timeframe.H1
    
    def test_expiration_check(self):
        """Test cache expiration logic."""
        # Create metadata from 2 hours ago
        two_hours_ago = datetime.now(timezone.utc).replace(microsecond=0) - pd.Timedelta(hours=2)
        
        metadata = CacheMetadata(
            request_hash="test_hash",
            symbol="EUR/USD",
            source=DataSource.FOREX_COM,
            timeframe=Timeframe.H1,
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
            bars_count=24,
            cached_at=two_hours_ago,
            file_path="/path/to/cache.parquet",
            file_size_bytes=1024
        )
        
        # Should not be expired with 24 hour max age
        assert not metadata.is_expired(max_age_hours=24.0)
        
        # Should be expired with 1 hour max age
        assert metadata.is_expired(max_age_hours=1.0)


@pytest.mark.unit
class TestQlibDataSpec:
    """Test QlibDataSpec Pydantic model."""
    
    def test_valid_spec_creation(self):
        """Test creating valid qlib data spec."""
        spec = QlibDataSpec(
            symbol="EUR/USD",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
            timeframe=Timeframe.H1,
            qlib_dir="/path/to/qlib"
        )
        
        assert spec.symbol == "EUR/USD"
        assert spec.timeframe == Timeframe.H1
    
    def test_path_generation(self):
        """Test qlib path generation methods."""
        spec = QlibDataSpec(
            symbol="EUR/USD",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
            timeframe=Timeframe.H1,
            qlib_dir="/path/to/qlib"
        )
        
        symbol_dir = spec.get_qlib_symbol_dir()
        assert symbol_dir == "/path/to/qlib/EUR_USD"  # Fixed: should use underscore
        
        file_path = spec.get_qlib_file_path()
        assert file_path == "/path/to/qlib/EUR_USD/1h.bin"  # Fixed: should use underscore
    
    def test_path_generation_various_symbols(self):
        """Test qlib path generation for various symbol formats."""
        test_cases = [
            ("EUR/USD", "/path/to/qlib/EUR_USD"),
            ("GBP/JPY", "/path/to/qlib/GBP_JPY"),
            ("BTC/USD", "/path/to/qlib/BTC_USD"),
            ("AAPL", "/path/to/qlib/AAPL"),  # No slash to replace
            ("NASDAQ:GOOGL", "/path/to/qlib/NASDAQ:GOOGL"),  # Different separator
        ]
        
        for symbol, expected_dir in test_cases:
            spec = QlibDataSpec(
                symbol=symbol,
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
                timeframe=Timeframe.H1,
                qlib_dir="/path/to/qlib"
            )
            
            symbol_dir = spec.get_qlib_symbol_dir()
            assert symbol_dir == expected_dir, f"Failed for symbol {symbol}"
            
            expected_file = f"{expected_dir}/1h.bin"
            file_path = spec.get_qlib_file_path()
            assert file_path == expected_file, f"Failed file path for symbol {symbol}" 