"""Abstract base calendar interface for market trading sessions.

This provides a modular foundation for different asset class calendars
that can be easily swapped without changing core trading logic.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Tuple

from ...types import MarketSession


class BaseCalendar(ABC):
    """Abstract base class for market calendars."""

    def __init__(self, timezone: str = "UTC"):
        """Initialize calendar with timezone.
        
        Args:
            timezone: Timezone for market operations (default: UTC)

        """
        self.timezone = timezone

    @abstractmethod
    def is_market_open(self, timestamp: datetime) -> bool:
        """Check if market is open at given timestamp.
        
        Args:
            timestamp: Datetime to check
            
        Returns:
            True if market is open, False otherwise

        """
        pass

    @abstractmethod
    def get_market_session(self, timestamp: datetime) -> MarketSession:
        """Get market session type at given timestamp.
        
        Args:
            timestamp: Datetime to check
            
        Returns:
            Market session type

        """
        pass

    @abstractmethod
    def get_trading_hours(self, date: datetime) -> List[Tuple[datetime, datetime]]:
        """Get trading hours for a specific date.
        
        Args:
            date: Date to get trading hours for
            
        Returns:
            List of (start_time, end_time) tuples for trading sessions

        """
        pass

    @abstractmethod
    def is_holiday(self, date: datetime) -> bool:
        """Check if given date is a market holiday.
        
        Args:
            date: Date to check
            
        Returns:
            True if it's a holiday, False otherwise

        """
        pass

    @abstractmethod
    def next_market_open(self, timestamp: datetime) -> datetime:
        """Get next market open time after given timestamp.
        
        Args:
            timestamp: Reference timestamp
            
        Returns:
            Next market open datetime

        """
        pass

    @abstractmethod
    def next_market_close(self, timestamp: datetime) -> datetime:
        """Get next market close time after given timestamp.
        
        Args:
            timestamp: Reference timestamp
            
        Returns:
            Next market close datetime

        """
        pass

    def is_weekend(self, date: datetime) -> bool:
        """Check if given date is a weekend.
        
        Args:
            date: Date to check
            
        Returns:
            True if weekend, False otherwise

        """
        return date.weekday() >= 5  # Saturday=5, Sunday=6

    def get_market_status(self, timestamp: datetime) -> dict:
        """Get comprehensive market status information.
        
        Args:
            timestamp: Datetime to check
            
        Returns:
            Dictionary with market status details

        """
        return {
            "timestamp": timestamp,
            "is_open": self.is_market_open(timestamp),
            "session": self.get_market_session(timestamp),
            "is_holiday": self.is_holiday(timestamp),
            "is_weekend": self.is_weekend(timestamp),
            "next_open": self.next_market_open(timestamp),
            "next_close": self.next_market_close(timestamp),
            "trading_hours": self.get_trading_hours(timestamp)
        }

    def filter_trading_times(self, timestamps: List[datetime]) -> List[datetime]:
        """Filter timestamps to only include trading hours.
        
        Args:
            timestamps: List of timestamps to filter
            
        Returns:
            Filtered list containing only trading hour timestamps

        """
        return [ts for ts in timestamps if self.is_market_open(ts)]
