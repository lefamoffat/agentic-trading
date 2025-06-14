import pandas as pd
import pytest
from scripts.data.download_historical import _prepare_for_qlib


def test_prepare_for_qlib():
    """
    Tests that the _prepare_for_qlib function correctly formats a DataFrame.
    """
    # 1. Create a sample DataFrame that mimics the output of DataProcessor
    # It should have timezone-aware timestamps and standardized column names.
    data = {
        "timestamp": pd.to_datetime(
            ["2024-01-01 10:00:00", "2024-01-01 11:00:00"]
        ).tz_localize("UTC"),
        "open": [1.1000, 1.1010],
        "high": [1.1020, 1.1030],
        "low": [1.0990, 1.1000],
        "close": [1.1010, 1.1020],
        "volume": [1000, 1200],
    }
    input_df = pd.DataFrame(data)

    # 2. Process the DataFrame with the function
    qlib_df = _prepare_for_qlib(input_df)

    # 3. Assert that the output is correctly formatted for Qlib
    
    # Check that the index is a DatetimeIndex and is named 'date'
    assert isinstance(qlib_df.index, pd.DatetimeIndex)
    assert qlib_df.index.name == "date"

    # CRITICAL: Check that the timezone has been removed (is now None)
    assert qlib_df.index.tz is None

    # Check for the presence of all required columns
    expected_columns = ["open", "high", "low", "close", "volume", "factor"]
    assert all(col in qlib_df.columns for col in expected_columns)
    
    # Check that the 'factor' column was added and has the correct value
    assert "factor" in qlib_df.columns
    assert (qlib_df["factor"] == 1.0).all()

    # Check that the data values are preserved
    assert qlib_df["open"].iloc[0] == 1.1000
    assert qlib_df["volume"].iloc[1] == 1200 