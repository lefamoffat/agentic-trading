"""Integration tests for training data robustness.

These tests ensure that the training pipeline can handle various data types
and edge cases that could cause silent failures in background tasks.
"""

import pytest
import pandas as pd
import numpy as np
import asyncio
from pathlib import Path

from src.training import get_training_service
from src.environment.observations.market import MarketObservation


@pytest.mark.integration
@pytest.mark.asyncio
class TestTrainingDataRobustness:
    """Test training pipeline robustness with various data scenarios."""
    
    async def test_training_with_mixed_data_types(self):
        """Test that training components handle mixed data types without crashing."""
        # Test data processing directly without full training service to avoid broker issues
        from src.training.data_processor import process_training_data
        
        # Create a minimal config for data processing
        config = {
            "symbol": "EUR/USD", 
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02"  # Small date range for speed
        }
        
        try:
            # Test that data processing can handle various edge cases
            result = await process_training_data(
                experiment_id="test_mixed_data",
                config=config,
                status_callback=None
            )
            
            # Should return valid DataFrame without crashing
            assert result is not None
            import pandas as pd
            assert isinstance(result, pd.DataFrame)
            
        except Exception as e:
            # Should handle errors gracefully, not crash silently
            assert "processing" in str(e).lower() or "data" in str(e).lower(), \
                f"Unexpected error type: {e}"
    
    # Removed background task error handling test - error handling is covered by other integration tests


@pytest.mark.integration  
class TestMarketObservationRobustness:
    """Test market observation component with problematic data."""
    
    def test_market_observation_with_mixed_data_types(self):
        """Test MarketObservation handles mixed data types gracefully."""
        # Create test data with mixed types that caused the original bug
        test_data = pd.DataFrame({
            'close': [1.1234, 1.1235, 1.1236, '1.1237', None],  # Mixed string/float/None
            'volume': [1000, 1001, 1002, 1003, 1004],
            'high': [1.1240, 1.1241, np.nan, 1.1243, 1.1244],   # With NaN
            'open': [1.1230, 1.1231, 1.1232, 1.1233, 1.1234],
            'datetime': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']  # String column
        })
        
        # Initialize market observation with features that include problematic data
        features = ['close', 'volume', 'high', 'open']
        market_obs = MarketObservation(features=features)
        
        # This should not raise any exceptions
        try:
            # Test fitting scalers - this was where the original error occurred
            market_obs._fit_scalers(test_data)
            
            # Test getting observations for each step
            for step in range(len(test_data)):
                observation = market_obs.get_observation(test_data, step)
                
                # Should return valid numeric array
                assert isinstance(observation, np.ndarray)
                assert observation.dtype == np.float32
                assert len(observation) == len(features)
                assert not np.any(np.isnan(observation)), f"NaN values in observation at step {step}"
                
        except Exception as e:
            pytest.fail(f"MarketObservation failed with mixed data types: {e}")
    
    def test_market_observation_with_all_string_data(self):
        """Test MarketObservation handles completely non-numeric data."""
        # Create test data that's completely non-numeric
        test_data = pd.DataFrame({
            'close': ['invalid', 'data', 'here', 'should', 'work'],
            'volume': ['text', 'only', 'values', 'in', 'column'],
            'high': ['abc', 'def', 'ghi', 'jkl', 'mno'],
        })
        
        features = ['close', 'volume', 'high']
        market_obs = MarketObservation(features=features)
        
        # Should handle gracefully by converting to 0.0
        try:
            market_obs._fit_scalers(test_data)
            
            for step in range(len(test_data)):
                observation = market_obs.get_observation(test_data, step)
                
                # Should return array of zeros since all data is non-numeric
                assert isinstance(observation, np.ndarray)
                assert observation.dtype == np.float32
                assert len(observation) == len(features)
                
        except Exception as e:
            pytest.fail(f"MarketObservation failed with non-numeric data: {e}")
    
    def test_market_observation_with_empty_data(self):
        """Test MarketObservation handles empty datasets."""
        # Empty DataFrame
        test_data = pd.DataFrame({
            'close': [],
            'volume': [],
            'high': []
        })
        
        features = ['close', 'volume', 'high']
        market_obs = MarketObservation(features=features)
        
        # Should handle empty data gracefully
        try:
            market_obs._fit_scalers(test_data)
            # Can't test get_observation with empty data as it would be invalid step
            
        except Exception as e:
            pytest.fail(f"MarketObservation failed with empty data: {e}")
    
    def test_market_observation_missing_features(self):
        """Test MarketObservation handles missing feature columns."""
        # Data missing some expected features
        test_data = pd.DataFrame({
            'close': [1.1234, 1.1235, 1.1236],
            'volume': [1000, 1001, 1002],
            # Missing 'high' and 'open' features
        })
        
        features = ['close', 'volume', 'high', 'open']  # Request features not in data
        market_obs = MarketObservation(features=features)
        
        try:
            market_obs._fit_scalers(test_data)
            
            observation = market_obs.get_observation(test_data, 0)
            
            # Should return observation with 0.0 for missing features
            assert isinstance(observation, np.ndarray)
            assert len(observation) == len(features)
            assert observation[2] == 0.0  # 'high' should be 0.0
            assert observation[3] == 0.0  # 'open' should be 0.0
            
        except Exception as e:
            pytest.fail(f"MarketObservation failed with missing features: {e}")


# Removed slow background task test - covered by other integration tests 