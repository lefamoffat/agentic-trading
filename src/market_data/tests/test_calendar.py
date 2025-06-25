"""Unit tests for ForexCalendar trading hours logic."""

from datetime import datetime, timezone

import pytest

from src.market_data.calendars.forex import ForexCalendar
from src.types import MarketSession

UTC = timezone.utc

class TestForexCalendar:
    """Validate core calendar open/close rules."""

    calendar = ForexCalendar()

    @pytest.mark.parametrize(
        "ts,expected",
        [
            # Friday 21:00 UTC (before close) => open
            (datetime(2023, 1, 6, 21, 0, tzinfo=UTC), True),
            # Friday 23:00 UTC => closed
            (datetime(2023, 1, 6, 23, 0, tzinfo=UTC), False),
            # Saturday 12:00 UTC => closed
            (datetime(2023, 1, 7, 12, 0, tzinfo=UTC), False),
            # Sunday 21:30 UTC => closed (opens 22:00)
            (datetime(2023, 1, 8, 21, 30, tzinfo=UTC), False),
            # Monday 01:00 UTC => open
            (datetime(2023, 1, 9, 1, 0, tzinfo=UTC), True),
        ],
    )
    def test_is_market_open(self, ts, expected):
        assert self.calendar.is_market_open(ts) is expected

    def test_get_market_session_overlap(self):
        # 12:30 UTC overlaps London/NewYork
        ts = datetime(2023, 1, 10, 12, 30, tzinfo=UTC)
        assert self.calendar.get_market_session(ts) == MarketSession.OVERLAP

    def test_next_market_open_and_close_monotonic(self):
        ts = datetime(2023, 1, 6, 23, 30, tzinfo=UTC)  # Friday after close
        next_open = self.calendar.next_market_open(ts)
        assert next_open > ts
        # From next open, the next close should be after open
        next_close = self.calendar.next_market_close(next_open)
        assert next_close > next_open 