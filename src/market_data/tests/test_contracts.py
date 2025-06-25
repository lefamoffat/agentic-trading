"""Unit tests for market_data.contracts module."""

from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest

from src.market_data.contracts import MarketDataRequest, MarketDataResponse
from src.types import DataSource, Timeframe

UTC = timezone.utc

class TestMarketDataRequest:
    """Validate input sanitisation and edge-case behaviour."""

    def test_symbol_is_uppercase_and_stripped(self):
        req = MarketDataRequest(
            symbol=" eur/usd ",
            source=DataSource.FOREX_COM,
            timeframe=Timeframe.H1,
            start_date=datetime(2023, 1, 1, tzinfo=UTC),
            end_date=datetime(2023, 1, 2, tzinfo=UTC),
        )
        assert req.symbol == "EUR/USD"

    def test_timezone_validation(self):
        with pytest.raises(ValueError):
            MarketDataRequest(
                symbol="EUR/USD",
                source=DataSource.FOREX_COM,
                timeframe=Timeframe.H1,
                start_date=datetime(2023, 1, 1),  # Naive
                end_date=datetime(2023, 1, 2, tzinfo=UTC),
            )

    def test_end_date_after_start_date_validation(self):
        with pytest.raises(ValueError):
            MarketDataRequest(
                symbol="EUR/USD",
                source=DataSource.FOREX_COM,
                timeframe=Timeframe.H1,
                start_date=datetime(2023, 1, 2, tzinfo=UTC),
                end_date=datetime(2023, 1, 1, tzinfo=UTC),
            )

class TestMarketDataResponse:
    """Test helper methods that compute coverage of the requested range."""

    @staticmethod
    def _make_df(num: int = 5, start: datetime | None = None, minutes: int = 60):
        start = start or datetime(2023, 1, 1, tzinfo=UTC)
        records = []
        for i in range(num):
            ts = start + timedelta(minutes=i * minutes)
            records.append({
                "timestamp": ts,
                "open": 1.0 + i * 0.1,
                "high": 1.0 + i * 0.1,
                "low": 1.0 + i * 0.1,
                "close": 1.0 + i * 0.1,
                "volume": 100,
            })
        return pd.DataFrame(records)

    def test_coverage_computation(self):
        start = datetime(2023, 1, 1, tzinfo=UTC)
        end = start + timedelta(hours=4)  # 4 hours requested
        request = MarketDataRequest(
            symbol="EUR/USD",
            source=DataSource.FOREX_COM,
            timeframe=Timeframe.H1,
            start_date=start,
            end_date=end,
        )
        df = self._make_df(num=5, start=start)  # 4 intervals = 4h coverage (5 rows)
        response = MarketDataResponse(
            request=request,
            data=df,
            bars_count=len(df),
            actual_start_date=df['timestamp'].min(),
            actual_end_date=df['timestamp'].max(),
        )
        coverage = response.get_date_range_coverage()
        assert 0.99 <= coverage <= 1.01  # Should be ~1.0
        assert response.is_complete_coverage()

    def test_missing_dataframe_columns_validation(self):
        start = datetime(2023, 1, 1, tzinfo=UTC)
        end = start + timedelta(hours=1)
        request = MarketDataRequest(
            symbol="EUR/USD",
            source=DataSource.FOREX_COM,
            timeframe=Timeframe.H1,
            start_date=start,
            end_date=end,
        )
        bad_df = pd.DataFrame({"timestamp": [start], "open": [1.0]})  # Missing required cols
        with pytest.raises(ValueError):
            MarketDataResponse(
                request=request,
                data=bad_df,
                bars_count=1,
                actual_start_date=start,
                actual_end_date=end,
            ) 