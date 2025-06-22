"""Forex market calendar implementation.

Handles 24/5 forex trading hours with major session identification
and overlap periods for optimal trading times.
"""

from datetime import datetime, time, timedelta, timezone
from typing import ClassVar, List, Tuple

import pytz

from src.types import MarketSession
from src.market_data.calendars.base import BaseCalendar

class ForexCalendar(BaseCalendar):
    """Forex market calendar implementation.

    Forex markets trade 24/5 from Sunday 17:00 EST to Friday 17:00 EST.
    Major sessions: Sydney, Tokyo, London, New York with overlap periods.
    """

    # Major forex trading sessions (all times in UTC)
    SESSIONS: ClassVar[dict[str, dict[str, time]]] = {
        "sydney": {"start": time(21, 0), "end": time(6, 0)},      # 21:00-06:00 UTC
        "tokyo": {"start": time(0, 0), "end": time(9, 0)},        # 00:00-09:00 UTC
        "london": {"start": time(7, 0), "end": time(16, 0)},      # 07:00-16:00 UTC
        "new_york": {"start": time(12, 0), "end": time(21, 0)}    # 12:00-21:00 UTC
    }

    # Major overlap periods (highest volatility)
    OVERLAPS: ClassVar[dict[str, dict[str, time]]] = {
        "london_ny": {"start": time(12, 0), "end": time(16, 0)},  # 12:00-16:00 UTC
        "sydney_tokyo": {"start": time(0, 0), "end": time(6, 0)}, # 00:00-06:00 UTC
        "tokyo_london": {"start": time(7, 0), "end": time(9, 0)}  # 07:00-09:00 UTC
    }

    # Forex holidays (when major markets are closed)
    FOREX_HOLIDAYS: ClassVar[list[str]] = [
        "01-01",  # New Year's Day
        "12-25",  # Christmas Day
    ]

    def __init__(self, timezone: str = "UTC"):
        """Initialize forex calendar.

        Args:
            timezone: Timezone for operations (default: UTC)

        """
        super().__init__(timezone)
        self.tz = pytz.timezone(timezone)

    def is_market_open(self, timestamp: datetime) -> bool:
        """Check if forex market is open.

        Forex is open 24/5 from Sunday 17:00 EST to Friday 17:00 EST.

        Args:
            timestamp: Datetime to check

        Returns:
            True if market is open, False otherwise

        """
        # Convert to UTC if needed
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        utc_time = timestamp.astimezone(timezone.utc)

        # Check if it's a holiday
        if self.is_holiday(utc_time):
            return False

        # Forex market times: Sunday 22:00 UTC to Friday 22:00 UTC
        weekday = utc_time.weekday()  # Monday=0, Sunday=6
        current_time = utc_time.time()

        # Friday after 22:00 UTC - market closed
        if weekday == 4 and current_time >= time(22, 0):
            return False

        # Saturday - market closed
        if weekday == 5:
            return False

        # Sunday before 22:00 UTC - market closed
        if weekday == 6 and current_time < time(22, 0):
            return False

        # All other times - market open
        return True

    def get_market_session(self, timestamp: datetime) -> MarketSession:
        """Get current forex session and check for overlaps.

        Args:
            timestamp: Datetime to check

        Returns:
            Market session type

        """
        if not self.is_market_open(timestamp):
            return MarketSession.CLOSED

        # Convert to UTC for session checking
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        utc_time = timestamp.astimezone(timezone.utc).time()

        # Check for overlap periods first (highest priority)
        for _overlap_name, overlap_times in self.OVERLAPS.items():
            if self._is_time_in_session(utc_time, overlap_times):
                return MarketSession.OVERLAP

        # Check individual sessions
        for _session_name, session_times in self.SESSIONS.items():
            if self._is_time_in_session(utc_time, session_times):
                return MarketSession.OPEN

        return MarketSession.OPEN  # Default to open if market is open

    def _is_time_in_session(self, current_time: time, session: dict) -> bool:
        """Check if current time falls within a trading session.

        Handles sessions that cross midnight (e.g., Sydney session).

        Args:
            current_time: Time to check
            session: Session dict with 'start' and 'end' times

        Returns:
            True if time is in session, False otherwise

        """
        start_time = session["start"]
        end_time = session["end"]

        if start_time <= end_time:
            # Normal session (doesn't cross midnight)
            return start_time <= current_time <= end_time
        else:
            # Session crosses midnight
            return current_time >= start_time or current_time <= end_time

    def get_trading_hours(self, date: datetime) -> List[Tuple[datetime, datetime]]:
        """Get forex trading hours for a specific date.

        Args:
            date: Date to get trading hours for

        Returns:
            List of (start_time, end_time) tuples for trading sessions

        """
        # Forex is essentially 24/5, so return the full day if market is open
        date_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        date_end = date.replace(hour=23, minute=59, second=59, microsecond=999999)

        if self.is_market_open(date_start):
            return [(date_start, date_end)]
        else:
            return []

    def is_holiday(self, date: datetime) -> bool:
        """Check if given date is a forex holiday.

        Args:
            date: Date to check

        Returns:
            True if it's a holiday, False otherwise

        """
        date_str = date.strftime("%m-%d")
        return date_str in self.FOREX_HOLIDAYS

    def next_market_open(self, timestamp: datetime) -> datetime:
        """Get next forex market open time.

        Args:
            timestamp: Reference timestamp

        Returns:
            Next market open datetime

        """
        current = timestamp

        # Check every hour until we find an open time
        for _ in range(24 * 7):  # Max 1 week search
            current += timedelta(hours=1)
            if self.is_market_open(current):
                return current

        # If we can't find an open time in a week, something's wrong
        raise ValueError("Could not find next market open time within 1 week")

    def next_market_close(self, timestamp: datetime) -> datetime:
        """Get next forex market close time.

        Args:
            timestamp: Reference timestamp

        Returns:
            Next market close datetime

        """
        current = timestamp

        # Check every hour until we find a closed time
        for _ in range(24 * 7):  # Max 1 week search
            current += timedelta(hours=1)
            if not self.is_market_open(current):
                return current

        # If we can't find a close time in a week, something's wrong
        raise ValueError("Could not find next market close time within 1 week")

    def get_active_session(self, timestamp: datetime) -> str:
        """Get the name of the currently active major session.

        Args:
            timestamp: Datetime to check

        Returns:
            Name of active session or "multiple" for overlaps

        """
        if not self.is_market_open(timestamp):
            return "closed"

        # Convert to UTC for session checking
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        utc_time = timestamp.astimezone(timezone.utc).time()

        active_sessions = []
        for session_name, session_times in self.SESSIONS.items():
            if self._is_time_in_session(utc_time, session_times):
                active_sessions.append(session_name)

        if len(active_sessions) == 0:
            return "none"
        elif len(active_sessions) == 1:
            return active_sessions[0]
        else:
            return "multiple"  # Overlap period

    def is_high_volatility_period(self, timestamp: datetime) -> bool:
        """Check if timestamp falls during high volatility overlap periods.

        Args:
            timestamp: Datetime to check

        Returns:
            True if during overlap period, False otherwise

        """
        return self.get_market_session(timestamp) == MarketSession.OVERLAP 