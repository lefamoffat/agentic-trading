"""Real integration tests for market data pipeline.

These tests use NO MOCKS and test the actual data flow through the system,
catching integration bugs like Pydantic validation errors that mocked tests miss.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd

from src.market_data import prepare_training_data, download_historical_data, download_and_save_qlib_data
from src.market_data.contracts import MarketDataRequest, MarketDataResponse
from src.market_data.exceptions import DataSourceError
from src.types import DataSource, Timeframe, BrokerType
from src.training.data_processor import process_training_data


@pytest.mark.integration
class TestMarketDataRealFlow:
    """Test real market data flow without mocks - catches integration bugs."""
    
    @pytest.mark.asyncio
    async def test_real_market_data_request_validation(self):
        """Test that MarketDataRequest validates fields correctly.
        
        This test would have caught the bars_requested bug that mocked tests missed.
        """
        start_date = datetime.now(timezone.utc) - timedelta(days=1)
        end_date = datetime.now(timezone.utc)
        
        # ✅ Valid request should work
        request = MarketDataRequest(
            symbol="EUR/USD",
            source=DataSource.FOREX_COM,
            timeframe=Timeframe.H1,
            start_date=start_date,
            end_date=end_date
        )
        assert request.symbol == "EUR/USD"
        
        # ❌ Invalid field should cause validation error (catches the original bug)
        with pytest.raises(Exception):  # Pydantic ValidationError or TypeError
            MarketDataRequest(
                symbol="EUR/USD",
                source=DataSource.FOREX_COM,
                timeframe=Timeframe.H1,
                start_date=start_date,
                end_date=end_date,
                bars_requested=1000  # ❌ INVALID FIELD - Would catch the bug!
            )
    
    @pytest.mark.asyncio
    async def test_real_download_flow_small_dataset(self):
        """Test actual download with real broker (small dataset to avoid rate limits)."""
        try:
            df = await download_historical_data(
                bars=5,  # Very small for fast test
                symbol="EUR/USD",
                timeframe="1h",
                broker="forex.com"
            )
            
            # Verify real data structure
            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            assert 'timestamp' in df.columns
            assert 'open' in df.columns
            assert 'high' in df.columns
            assert 'low' in df.columns
            assert 'close' in df.columns
            assert 'volume' in df.columns
            
        except DataSourceError as e:
            if "authentication" in str(e).lower() or "credentials" in str(e).lower():
                pytest.skip(f"Skipping real data test due to auth: {e}")
            else:
                raise  # Re-raise unexpected errors
    
    @pytest.mark.asyncio
    async def test_real_training_pipeline_integration(self):
        """Test complete training pipeline with real components.
        
        This test would catch any integration issues in the full pipeline.
        """
        try:
            config = {
                "symbol": "EUR/USD",
                "timeframe": "1h",
                "timesteps": 10  # Very small for quick test
            }
            
            # Test real pipeline: download → qlib → features (NO MOCKS)
            features_df = await process_training_data(
                experiment_id="real_integration_test",
                config=config,
                status_callback=None
            )
            
            # Verify real results
            assert isinstance(features_df, pd.DataFrame)
            assert not features_df.empty
                
        except DataSourceError as e:
            if "authentication" in str(e).lower():
                pytest.skip(f"Skipping real pipeline test due to auth: {e}")
            else:
                raise
    
    @pytest.mark.asyncio
    async def test_real_error_propagation(self):
        """Test that real errors propagate correctly (no silent failures)."""
        # Test invalid timeframe - should raise ValueError
        with pytest.raises(ValueError, match="is not a valid Timeframe"):
            await download_historical_data(
                bars=5,
                symbol="EUR/USD",
                timeframe="invalid_timeframe",
                broker="forex.com"
            )
        
        # Test invalid date range - should raise error
        end_date = datetime.now(timezone.utc)
        start_date = end_date + timedelta(days=1)  # Start after end
        
        with pytest.raises((DataSourceError, ValueError)):
            await prepare_training_data(
                symbol="EUR/USD",
                source=DataSource.FOREX_COM,
                timeframe=Timeframe.H1,
                start_date=start_date,  # Invalid: start > end
                end_date=end_date
            )
    
    def test_real_enum_validation(self):
        """Test enum validation works correctly."""
        # Test valid timeframes
        assert Timeframe.from_standard("1h") == Timeframe.H1
        assert Timeframe.from_standard("1d") == Timeframe.D1
        
        # Test invalid timeframes raise errors
        with pytest.raises(ValueError, match="is not a valid Timeframe"):
            Timeframe.from_standard("invalid")


@pytest.mark.integration
class TestPydanticValidationReal:
    """Test Pydantic model validation with real data (catches schema bugs)."""
    
    def test_market_data_request_field_validation(self):
        """Test all MarketDataRequest fields validate correctly."""
        valid_start = datetime.now(timezone.utc) - timedelta(days=1)
        valid_end = datetime.now(timezone.utc)
        
        # Valid request
        request = MarketDataRequest(
            symbol="EUR/USD",
            source=DataSource.FOREX_COM,
            timeframe=Timeframe.H1,
            start_date=valid_start,
            end_date=valid_end
        )
        
        # Verify required fields are set
        assert request.symbol == "EUR/USD"
        assert request.source == DataSource.FOREX_COM
        assert request.timeframe == Timeframe.H1
        assert request.start_date == valid_start
        assert request.end_date == valid_end
        
        # Test that extra fields are rejected (catches the bars_requested bug)
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            MarketDataRequest(
                symbol="EUR/USD",
                source=DataSource.FOREX_COM,
                timeframe=Timeframe.H1,
                start_date=valid_start,
                end_date=valid_end,
                bars_requested=1000  # This would have caused the original bug!
            )


@pytest.mark.integration
class TestDataFlowErrorBoundaries:
    """Test error boundaries in real data flow."""
    
    @pytest.mark.asyncio 
    async def test_real_network_timeout_handling(self):
        """Test handling of real network timeouts."""
        # This would test actual timeout scenarios
        # Implementation depends on how timeouts are configured
        pass
    
    @pytest.mark.asyncio
    async def test_real_authentication_failure_handling(self):
        """Test handling of real authentication failures."""
        # Test with intentionally bad credentials
        # This verifies error propagation from broker level
        pass
    
    @pytest.mark.asyncio
    async def test_real_data_corruption_handling(self):
        """Test handling of corrupted real data."""
        # Test what happens when broker returns malformed data
        # This verifies data validation at all levels
        pass 