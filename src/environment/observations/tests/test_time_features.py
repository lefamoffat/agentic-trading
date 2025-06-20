#!/usr/bin/env python3
"""Unit tests for TimeObservation component."""
import pytest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.environment.observations.time_features import TimeObservation


class TestTimeObservation:
    """Test suite for TimeObservation component."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.time_obs = TimeObservation(
            trading_start_hour=9,
            trading_end_hour=17,
            trading_timezone="UTC",
            exclude_weekends=True
        )
    
    @pytest.mark.unit
    def test_init_with_defaults(self):
        """Test TimeObservation initialization with default parameters."""
        obs = TimeObservation()
        
        assert obs.trading_start_hour == 7
        assert obs.trading_end_hour == 17
        assert obs.trading_timezone == "UTC"
        assert obs.exclude_weekends is True
        assert obs.include_market_session is True
        assert obs.include_time_of_day is True
        assert obs.include_day_of_week is True
    
    @pytest.mark.unit
    def test_init_with_custom_parameters(self):
        """Test TimeObservation initialization with custom parameters."""
        obs = TimeObservation(
            trading_start_hour=22,
            trading_end_hour=6,
            exclude_weekends=False,
            include_market_session=False
        )
        
        assert obs.trading_start_hour == 22
        assert obs.trading_end_hour == 6
        assert obs.exclude_weekends is False
        assert obs.include_market_session is False
    
    @pytest.mark.unit
    def test_observation_size_all_features(self):
        """Test observation size with all features enabled."""
        assert self.time_obs.observation_size == 9  # 1 + 1 + 7
    
    @pytest.mark.unit
    def test_observation_size_partial_features(self):
        """Test observation size with only some features enabled."""
        obs = TimeObservation(
            include_market_session=True,
            include_time_of_day=False,
            include_day_of_week=False
        )
        assert obs.observation_size == 1  # Only market session
        
        obs = TimeObservation(
            include_market_session=False,
            include_time_of_day=True,
            include_day_of_week=True
        )
        assert obs.observation_size == 8  # 1 + 7
    
    @pytest.mark.unit
    def test_feature_names_all_enabled(self):
        """Test feature names with all features enabled."""
        feature_names = self.time_obs.get_feature_names()
        
        expected_names = [
            "market_open",
            "time_of_day_normalized", 
            "day_monday", "day_tuesday", "day_wednesday", "day_thursday",
            "day_friday", "day_saturday", "day_sunday"
        ]
        
        assert feature_names == expected_names
        assert len(feature_names) == 9
    
    @pytest.mark.unit
    def test_feature_names_partial_features(self):
        """Test feature names with partial features enabled."""
        obs = TimeObservation(
            include_market_session=True,
            include_time_of_day=False,
            include_day_of_week=False
        )
        
        feature_names = obs.get_feature_names()
        assert feature_names == ["market_open"]
    
    @pytest.mark.unit
    def test_is_market_open_during_trading_hours(self):
        """Test market open detection during trading hours."""
        # Monday 10:00 UTC (within 9-17 trading hours)
        timestamp = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)  # Monday
        
        assert self.time_obs._is_market_open(timestamp) is True
    
    @pytest.mark.unit
    def test_is_market_closed_outside_trading_hours(self):
        """Test market closed detection outside trading hours."""
        # Monday 8:00 UTC (before 9:00 start)
        timestamp = datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc)
        
        assert self.time_obs._is_market_open(timestamp) is False
        
        # Monday 18:00 UTC (after 17:00 end)
        timestamp = datetime(2024, 1, 1, 18, 0, tzinfo=timezone.utc)
        
        assert self.time_obs._is_market_open(timestamp) is False
    
    @pytest.mark.unit
    def test_is_market_closed_on_weekends(self):
        """Test market closed detection on weekends."""
        # Saturday 10:00 UTC (would be open but weekend)
        timestamp = datetime(2024, 1, 6, 10, 0, tzinfo=timezone.utc)  # Saturday
        
        assert self.time_obs._is_market_open(timestamp) is False
        
        # Sunday 10:00 UTC
        timestamp = datetime(2024, 1, 7, 10, 0, tzinfo=timezone.utc)  # Sunday
        
        assert self.time_obs._is_market_open(timestamp) is False
    
    @pytest.mark.unit
    def test_is_market_open_weekends_allowed(self):
        """Test market open detection when weekends are allowed."""
        obs = TimeObservation(
            trading_start_hour=9,
            trading_end_hour=17,
            exclude_weekends=False
        )
        
        # Saturday 10:00 UTC (should be open when weekends allowed)
        timestamp = datetime(2024, 1, 6, 10, 0, tzinfo=timezone.utc)
        
        assert obs._is_market_open(timestamp) is True
    
    @pytest.mark.unit
    def test_overnight_trading_hours(self):
        """Test market hours that cross midnight (e.g., 22:00-6:00)."""
        obs = TimeObservation(
            trading_start_hour=22,
            trading_end_hour=6,
            exclude_weekends=True
        )
        
        # 23:00 should be open (evening session)
        timestamp = datetime(2024, 1, 1, 23, 0, tzinfo=timezone.utc)
        assert obs._is_market_open(timestamp) is True
        
        # 2:00 should be open (morning session)
        timestamp = datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc)
        assert obs._is_market_open(timestamp) is True
        
        # 10:00 should be closed
        timestamp = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        assert obs._is_market_open(timestamp) is False
    
    @pytest.mark.unit
    def test_normalized_time_of_day_normal_hours(self):
        """Test normalized time of day calculation for normal trading hours."""
        # 9:00 (start) should be 0.0
        timestamp = datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc)
        normalized_time = self.time_obs._get_normalized_time_of_day(timestamp)
        assert normalized_time == 0.0
        
        # 13:00 (middle) should be 0.5
        timestamp = datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc)
        normalized_time = self.time_obs._get_normalized_time_of_day(timestamp)
        assert normalized_time == 0.5
        
        # 17:00 (end) should be 1.0
        timestamp = datetime(2024, 1, 1, 17, 0, tzinfo=timezone.utc)
        normalized_time = self.time_obs._get_normalized_time_of_day(timestamp)
        assert normalized_time == 1.0
    
    @pytest.mark.unit
    def test_normalized_time_of_day_outside_hours(self):
        """Test normalized time of day calculation outside trading hours."""
        # 8:00 (before start) should be 0.0
        timestamp = datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc)
        normalized_time = self.time_obs._get_normalized_time_of_day(timestamp)
        assert normalized_time == 0.0
        
        # 18:00 (after end) should be 1.0
        timestamp = datetime(2024, 1, 1, 18, 0, tzinfo=timezone.utc)
        normalized_time = self.time_obs._get_normalized_time_of_day(timestamp)
        assert normalized_time == 1.0
    
    @pytest.mark.unit
    def test_get_observation_with_datetime_index(self):
        """Test observation generation with DataFrame having datetime index."""
        # Create test data with datetime index
        timestamps = pd.date_range('2024-01-01 10:00:00', periods=3, freq='1h')
        data = pd.DataFrame({
            'close': [1.1, 1.2, 1.3],
            'volume': [100, 200, 300]
        }, index=timestamps)
        
        obs = self.time_obs.get_observation(data, current_step=0)
        
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert len(obs) == 9  # All features enabled
        
        # First observation should be: market_open=1, time_of_day=0.125, Monday=1, others=0
        assert obs[0] == 1.0  # market_open (10:00 is within 9-17)
        assert 0.0 <= obs[1] <= 1.0  # time_of_day_normalized
        assert obs[2] == 1.0  # day_monday (2024-01-01 was Monday)
        assert np.sum(obs[2:9]) == 1.0  # Only one day should be 1.0
    
    @pytest.mark.unit
    def test_get_observation_with_timestamp_column(self):
        """Test observation generation with DataFrame having timestamp column."""
        data = pd.DataFrame({
            'timestamp': ['2024-01-01 10:00:00'],
            'close': [1.1],
            'volume': [100]
        })
        
        obs = self.time_obs.get_observation(data, current_step=0)
        
        assert isinstance(obs, np.ndarray)
        assert len(obs) == 9
        assert obs[0] == 1.0  # market_open
    
    @pytest.mark.unit
    def test_get_observation_fallback_to_current_time(self):
        """Test observation generation falls back to current time when no timestamp available."""
        data = pd.DataFrame({
            'close': [1.1],
            'volume': [100]
        })
        
        obs = self.time_obs.get_observation(data, current_step=0)
        
        assert isinstance(obs, np.ndarray)
        assert len(obs) == 9
        # Should not crash and should return valid observation
    
    @pytest.mark.unit
    def test_get_observation_different_days_of_week(self):
        """Test observation generation for different days of the week."""
        # Test each day of the week
        days = ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', 
                '2024-01-05', '2024-01-06', '2024-01-07']  # Mon-Sun
        
        for i, date_str in enumerate(days):
            timestamps = pd.date_range(f'{date_str} 10:00:00', periods=1, freq='1h')
            data = pd.DataFrame({'close': [1.1]}, index=timestamps)
            
            obs = self.time_obs.get_observation(data, current_step=0)
            
            # Check that the correct day is encoded
            day_encoding = obs[2:9]  # Days are features 2-8
            assert day_encoding[i] == 1.0  # This day should be 1.0
            assert np.sum(day_encoding) == 1.0  # Only one day should be active
    
    @pytest.mark.unit
    def test_edge_case_boundary_times(self):
        """Test edge cases at trading hour boundaries."""
        # Test exactly at start time
        timestamp = datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
        assert self.time_obs._is_market_open(timestamp) is True
        
        # Test one minute before start
        timestamp = datetime(2024, 1, 1, 8, 59, 59, tzinfo=timezone.utc)
        assert self.time_obs._is_market_open(timestamp) is False
        
        # Test exactly at end time
        timestamp = datetime(2024, 1, 1, 17, 0, 0, tzinfo=timezone.utc)
        assert self.time_obs._is_market_open(timestamp) is False
        
        # Test one minute before end
        timestamp = datetime(2024, 1, 1, 16, 59, 59, tzinfo=timezone.utc)
        assert self.time_obs._is_market_open(timestamp) is True
    
    @pytest.mark.unit
    def test_invalid_step_index(self):
        """Test error handling for invalid step indices."""
        data = pd.DataFrame({'close': [1.1]})
        
        # Should handle gracefully and not crash
        try:
            obs = self.time_obs.get_observation(data, current_step=5)  # Out of bounds
            assert isinstance(obs, np.ndarray)
        except (IndexError, KeyError):
            # Acceptable to raise error for invalid indices
            pass 