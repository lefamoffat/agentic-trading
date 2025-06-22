#!/usr/bin/env python3
"""Time-based observation component for trading environment.

This module provides time-based features that allow the RL agent to be aware of:
- Trading hours
- Day of week
- Market sessions
- Time-based constraints
"""
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd

class TimeObservation:
    """Handles time-based observations for trading awareness.
    
    This allows the RL agent to observe and respect trading hours,
    market sessions, and other time-based constraints.
    """
    
    def __init__(self, 
                 trading_start_hour: int = 7,
                 trading_end_hour: int = 17,
                 trading_timezone: str = "UTC",
                 exclude_weekends: bool = True,
                 include_market_session: bool = True,
                 include_time_of_day: bool = True,
                 include_day_of_week: bool = True):
        """Initialize time observation component.
        
        Args:
            trading_start_hour: Trading start hour (0-23)
            trading_end_hour: Trading end hour (0-23)
            trading_timezone: Trading timezone
            exclude_weekends: Whether weekends are excluded
            include_market_session: Include market session info
            include_time_of_day: Include normalized time of day
            include_day_of_week: Include day of week encoding
        """
        self.trading_start_hour = trading_start_hour
        self.trading_end_hour = trading_end_hour
        self.trading_timezone = trading_timezone
        self.exclude_weekends = exclude_weekends
        self.include_market_session = include_market_session
        self.include_time_of_day = include_time_of_day
        self.include_day_of_week = include_day_of_week
    
    def get_observation(self, data: pd.DataFrame, current_step: int) -> np.ndarray:
        """Get time-based observation for current step.
        
        Args:
            data: Market data DataFrame with timestamp/index
            current_step: Current step index
            
        Returns:
            Time-based features array
        """
        observations = []
        
        # Get current timestamp
        if hasattr(data.index, 'to_pydatetime'):
            # DatetimeIndex
            current_time = data.index[current_step]
        elif 'timestamp' in data.columns:
            current_time = pd.to_datetime(data.iloc[current_step]['timestamp'])
        else:
            # Fallback: use current datetime
            current_time = datetime.now(timezone.utc)
        
        # Ensure timezone awareness
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        
        # Convert to trading timezone if needed
        if self.trading_timezone != "UTC":
            # For now, assume UTC. Could extend with pytz for other timezones
            pass
        
        # Market session indicator (0 = closed, 1 = open)
        if self.include_market_session:
            is_market_open = self._is_market_open(current_time)
            observations.append(float(is_market_open))
        
        # Normalized time of day (0-1, where 0 = start of trading, 1 = end of trading)
        if self.include_time_of_day:
            time_of_day = self._get_normalized_time_of_day(current_time)
            observations.append(time_of_day)
        
        # Day of week encoding (Monday=0, Sunday=6)
        if self.include_day_of_week:
            # One-hot encoding for day of week (7 features)
            day_of_week = current_time.weekday()  # Monday=0, Sunday=6
            day_encoding = [0.0] * 7
            day_encoding[day_of_week] = 1.0
            observations.extend(day_encoding)
        
        return np.array(observations, dtype=np.float32)
    
    def _is_market_open(self, timestamp: datetime) -> bool:
        """Check if market is open at given timestamp.
        
        Args:
            timestamp: Timestamp to check
            
        Returns:
            True if market is open, False otherwise
        """
        # Check weekends
        if self.exclude_weekends and timestamp.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        # Check trading hours
        current_hour = timestamp.hour
        if self.trading_start_hour <= self.trading_end_hour:
            # Normal case: 7-17
            return self.trading_start_hour <= current_hour < self.trading_end_hour
        else:
            # Overnight case: 22-6 (crosses midnight)
            return current_hour >= self.trading_start_hour or current_hour < self.trading_end_hour
    
    def _get_normalized_time_of_day(self, timestamp: datetime) -> float:
        """Get normalized time of day within trading hours.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Normalized time (0.0 = trading start, 1.0 = trading end)
        """
        current_hour = timestamp.hour + timestamp.minute / 60.0
        
        if self.trading_start_hour <= self.trading_end_hour:
            # Normal trading hours
            trading_duration = self.trading_end_hour - self.trading_start_hour
            if current_hour < self.trading_start_hour:
                return 0.0  # Before trading starts
            elif current_hour >= self.trading_end_hour:
                return 1.0  # After trading ends
            else:
                return (current_hour - self.trading_start_hour) / trading_duration
        else:
            # Overnight trading hours (crosses midnight)
            if current_hour >= self.trading_start_hour:
                # Evening session
                evening_duration = 24 - self.trading_start_hour
                return (current_hour - self.trading_start_hour) / (evening_duration + self.trading_end_hour)
            else:
                # Morning session
                evening_duration = 24 - self.trading_start_hour
                return (evening_duration + current_hour) / (evening_duration + self.trading_end_hour)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names in observation order.
        
        Returns:
            List of feature names
        """
        features = []
        
        if self.include_market_session:
            features.append("market_open")
        
        if self.include_time_of_day:
            features.append("time_of_day_normalized")
        
        if self.include_day_of_week:
            days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            features.extend([f"day_{day}" for day in days])
        
        return features
    
    @property
    def observation_size(self) -> int:
        """Get size of observation vector.
        
        Returns:
            Number of features in observation
        """
        size = 0
        if self.include_market_session:
            size += 1
        if self.include_time_of_day:
            size += 1
        if self.include_day_of_week:
            size += 7  # One-hot encoding for 7 days
        return size 