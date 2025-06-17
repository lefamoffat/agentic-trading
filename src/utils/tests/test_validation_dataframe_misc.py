import pandas as pd
import pytest

from src.utils.exceptions import ValidationError
from src.utils.validation.dataframe import (
    check_data_gaps,
    validate_ohlcv_consistency,
    validate_ohlcv_data,
)


@pytest.fixture
def ohlcv_df():
    return pd.DataFrame({
        "open": [1.0, 1.1, 1.2],
        "high": [1.1, 1.2, 1.3],
        "low": [0.9, 1.0, 1.1],
        "close": [1.05, 1.15, 1.25],
        "volume": [100, 110, 120],
    })


@pytest.mark.unit
class TestValidateOhlcv:
    def test_validate_success(self, ohlcv_df):
        out = validate_ohlcv_data(ohlcv_df.copy())
        pd.testing.assert_frame_equal(out, ohlcv_df)

    def test_missing_columns(self):
        df = pd.DataFrame({"open": [1], "close": [1]})
        with pytest.raises(ValidationError):
            validate_ohlcv_data(df)

    def test_consistency_error(self, ohlcv_df):
        bad = ohlcv_df.copy()
        bad.loc[0, "high"] = 0.5  # high < low
        with pytest.raises(ValidationError):
            validate_ohlcv_consistency(bad)


@pytest.mark.unit
def test_check_data_gaps():
    ts = pd.date_range("2024-01-01", periods=5, freq="1H")
    df = pd.DataFrame({"timestamp": ts, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 10})
    gaps = check_data_gaps(df)
    assert gaps["total_gaps"] == 0
