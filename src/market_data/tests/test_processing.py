"""Tests for market data processing functionality."""

import pandas as pd
import pytest
from datetime import datetime, timezone

from src.market_data.processing import DataProcessor
from src.market_data.download import prepare_for_qlib

@pytest.mark.unit
class TestDataProcessor:
    """Test DataProcessor functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create DataProcessor for testing."""
        return DataProcessor(symbol="EUR/USD", asset_class="forex")
    
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
    
    def test_processor_initialization(self, processor):
        """Test DataProcessor initialization."""
        assert processor.symbol == "EUR/USD"
        assert processor.asset_class == "forex"
        assert processor.calendar is not None
    
    def test_standardize_dataframe_valid(self, processor, sample_dataframe):
        """Test DataFrame standardization with valid data."""
        result = processor.standardize_dataframe(sample_dataframe, broker_name="test")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 24
        assert list(result.columns) == processor.REQUIRED_COLUMNS
        assert result['timestamp'].dtype == 'object'  # Formatted as string
    
    def test_standardize_dataframe_empty(self, processor):
        """Test DataFrame standardization with empty data."""
        empty_df = pd.DataFrame()
        result = processor.standardize_dataframe(empty_df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == processor.REQUIRED_COLUMNS
    
    def test_validate_required_columns_missing(self, processor):
        """Test validation with missing required columns."""
        incomplete_df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            'open': [1.1000],
            # Missing high, low, close, volume
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            processor._validate_required_columns(incomplete_df)
    
    def test_validate_data_quality(self, processor, sample_dataframe):
        """Test data quality validation."""
        quality = processor.validate_data_quality(sample_dataframe)
        
        assert isinstance(quality, dict)
        assert "total_records" in quality
        assert "quality_score" in quality
        assert "issues" in quality
        assert quality["total_records"] == 24
        assert quality["quality_score"] >= 0.0

@pytest.mark.unit
def test_prepare_for_qlib():
    """Tests that the prepare_for_qlib function correctly formats a DataFrame."""
    data = {
        "timestamp": pd.to_datetime([
            "2024-01-01 10:00:00", "2024-01-01 11:00:00"
        ]).tz_localize("UTC"),
        "open": [1.1000, 1.1010],
        "high": [1.1020, 1.1030],
        "low": [1.0990, 1.1000],
        "close": [1.1010, 1.1020],
        "volume": [1000, 1200],
    }
    input_df = pd.DataFrame(data)

    qlib_df = prepare_for_qlib(input_df)

    assert isinstance(qlib_df.index, pd.DatetimeIndex)
    assert qlib_df.index.name == "date"
    assert qlib_df.index.tz is None

    expected_columns = ["open", "high", "low", "close", "volume", "factor"]
    assert all(col in qlib_df.columns for col in expected_columns)
    assert "factor" in qlib_df.columns
    assert (qlib_df["factor"] == 1.0).all()
    assert qlib_df["open"].iloc[0] == 1.1000
    assert qlib_df["volume"].iloc[1] == 1200 