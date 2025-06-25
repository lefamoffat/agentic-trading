"""Unit tests for DataProcessor standardization and validation."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.market_data.processing import DataProcessor

UTC = timezone.utc

class TestDataProcessor:
    """Test core data cleaning routines."""

    def _raw_df(self):
        return pd.DataFrame(
            [
                {
                    "timestamp": datetime(2023, 1, 1, 0, 0, tzinfo=UTC),
                    "open": 1.1000,
                    "high": 1.1050,
                    "low": 1.0950,
                    "close": 1.1020,
                    "volume": 1000,
                },
                {
                    # Intentionally set high below close to trigger fix
                    "timestamp": datetime(2023, 1, 1, 1, 0, tzinfo=UTC),
                    "open": 1.1020,
                    "high": 1.1010,  # invalid high
                    "low": 1.0900,
                    "close": 1.1030,
                    "volume": 1200,
                },
                {
                    # Duplicate timestamp to test deduplication
                    "timestamp": datetime(2023, 1, 1, 1, 0, tzinfo=UTC),
                    "open": 1.1021,
                    "high": 1.1060,
                    "low": 1.0980,
                    "close": 1.1040,
                    "volume": 1100,
                },
            ]
        )

    def test_standardize_dataframe_outputs_expected_columns_and_order(self):
        processor = DataProcessor(symbol="EUR/USD")
        standardized = processor.standardize_dataframe(self._raw_df())

        expected_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        assert list(standardized.columns) == expected_cols
        # All rows unique timestamps
        assert standardized['timestamp'].is_unique
        # Timestamps formatted as ISO strings ending with Z
        for ts in standardized['timestamp']:
            assert ts.endswith("Z")

    def test_validate_data_quality_detects_issues(self):
        processor = DataProcessor(symbol="EUR/USD")
        df = self._raw_df()
        # Inject zero price to create additional issue
        df.loc[0, 'close'] = 0.0
        standardized = processor.standardize_dataframe(df)
        report = processor.validate_data_quality(standardized)
        assert report['total_records'] == len(standardized)
        # Should detect at least one issue (zero prices or extreme moves)
        assert len(report['issues']) >= 1
        assert 0.0 <= report['quality_score'] <= 1.0 