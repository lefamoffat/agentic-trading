"""DataFrame and OHLCV data validation utilities.

This module provides validation functions for pandas DataFrames,
Series, OHLCV market data, and data quality assessment.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.exceptions import MissingDataError, ValidationError, create_context

def validate_columns_exist(df: pd.DataFrame,
                           required_columns: List[str],
                           name: str = "DataFrame") -> pd.DataFrame:
    """Validate that a DataFrame has the required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        name: Parameter name for error messages
    Returns:
        The validated DataFrame
    Raises:
        ValidationError: If validation fails

    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValidationError(
            f"Missing required columns for {name}: {missing_columns}",
            context=create_context(
                missing_columns=missing_columns, required=required_columns
            ),
        )
    return df

def validate_numeric_series(series: pd.Series,
                            allow_nan: bool = False,
                            name: str = "Series") -> pd.Series:
    """Validate that a Series contains numeric data.

    Args:
        series: Series to validate
        allow_nan: Whether to allow NaN values
        name: Parameter name for error messages
    Returns:
        The validated Series
    Raises:
        ValidationError: If validation fails

    """
    if not pd.api.types.is_numeric_dtype(series):
        raise ValidationError(
            f"{name} must contain numeric data, but found type {series.dtype}",
            context=create_context(parameter=name, dtype=str(series.dtype))
        )
    if not allow_nan and series.isnull().any():
        raise ValidationError(
            f"{name} must not contain NaN values",
            context=create_context(parameter=name)
        )
    return series

def validate_ohlcv_data(
    data: pd.DataFrame, required_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Validate OHLCV data format and integrity.

    Args:
        data: DataFrame with OHLCV data
        required_columns: List of required column names (default:
            ['open', 'high', 'low', 'close', 'volume'])

    Returns:
        Validated DataFrame
    Raises:
        ValidationError: If validation fails

    """
    if required_columns is None:
        required_columns = ['open', 'high', 'low', 'close', 'volume']

    # 1. Check for required columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValidationError(
            f"Missing required OHLCV columns: {missing_columns}",
            context=create_context(
                missing_columns=missing_columns, required=required_columns
            )
        )

    # 2. Check for numeric types in OHLCV columns
    for col in required_columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise ValidationError(
                f"Column '{col}' must be numeric, but found type {data[col].dtype}",
                context=create_context(column=col, dtype=str(data[col].dtype)),
            )

    # 3. Check for consistency (e.g., high >= low)
    if 'high' in data.columns and 'low' in data.columns:
        if (data['high'] < data['low']).any():
            raise ValidationError(
                "Data inconsistency: 'high' column contains values less "
                "than 'low' column",
                context=create_context(
                    inconsistent_rows=(data['high'] < data['low']).sum()
                ),
            )

    # 4. Check for non-negative prices and volume
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in data.columns and (data[col] < 0).any():
            raise ValidationError(
                f"Column '{col}' must not contain negative values",
                context=create_context(column=col)
            )

    return data

def validate_ohlcv_consistency(data: pd.DataFrame) -> pd.DataFrame:
    """Validate OHLCV data consistency (relationships between OHLC values).

    Args:
        data: DataFrame with OHLCV data
    Returns:
        Validated DataFrame
    Raises:
        ValidationError: If data is inconsistent

    """
    ohlc_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in ohlc_cols if col not in data.columns]
    if missing_cols:
        raise ValidationError(
            f"Missing OHLC columns for consistency check: {missing_cols}"
        )

    # Check for high < low condition
    if (data['high'] < data['low']).any():
        raise ValidationError(
            "Data inconsistency: 'high' contains values less than 'low'.",
            context=create_context(
                inconsistent_rows=(data['high'] < data['low']).sum()
            ),
        )

    # Check for high < open or high < close
    if (data['high'] < data['open']).any() or (data['high'] < data['close']).any():
        raise ValidationError(
            "Data inconsistency: 'high' is less than 'open' or 'close'.",
            context=create_context(
                high_lt_open=(data['high'] < data['open']).sum(),
                high_lt_close=(data['high'] < data['close']).sum(),
            ),
        )

    # Check for low > open or low > close
    if (data['low'] > data['open']).any() or (data['low'] > data['close']).any():
        raise ValidationError(
            "Data inconsistency: 'low' is greater than 'open' or 'close'.",
            context=create_context(
                low_gt_open=(data['low'] > data['open']).sum(),
                low_gt_close=(data['low'] > data['close']).sum(),
            ),
        )

    return data

def validate_data_completeness(data: pd.DataFrame,
                               min_rows: int = 100,
                               max_gap_percentage: float = 5.0) -> pd.DataFrame:
    """Validate data completeness requirements.

    Args:
        data: DataFrame to validate
        min_rows: Minimum number of rows required
        max_gap_percentage: Maximum allowed percentage of missing values
    Returns:
        Validated DataFrame
    Raises:
        MissingDataError: If data completeness requirements not met

    """
    if len(data) < min_rows:
        raise MissingDataError(
            f"Insufficient data: {len(data)} rows, minimum {min_rows} required",
            context=create_context(rows=len(data), min_required=min_rows)
        )

    nan_percentage = data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) * 100
    if nan_percentage > max_gap_percentage:
        raise ValidationError(
            f"Data contains {nan_percentage:.1f}% gaps, "
            f"maximum {max_gap_percentage}% allowed",
            context=create_context(
                nan_percentage=nan_percentage,
                max_allowed=max_gap_percentage
            )
        )
    return data

def validate_data_quality(
    data: pd.DataFrame, min_quality_score: float = 0.8
) -> pd.DataFrame:
    """Validate data quality and return the DataFrame if quality is acceptable.

    Args:
        data: DataFrame to validate
        min_quality_score: Minimum required quality score
    Returns:
        Validated DataFrame
    Raises:
        ValidationError: If quality score is below minimum

    """
    quality_score = calculate_data_quality_score(data)

    if quality_score < min_quality_score:
        raise ValidationError(
            f"Data quality score {quality_score:.2f} below minimum "
            f"{min_quality_score:.2f}",
            context=create_context(
                quality_score=quality_score, min_required=min_quality_score
            ),
        )

    return data

def calculate_data_quality_score(
    data: pd.DataFrame, weights: Optional[Dict[str, float]] = None
) -> float:
    """Calculate a data quality score based on completeness, consistency, and volume.
    The score is a weighted average of:
    - Completeness (40%): How many values are not NaN.
    - Consistency (40%): How many rows have valid OHLC (high >= low).
    - Volume Health (20%): How many rows have non-zero volume.

    Args:
        data: DataFrame to analyze.
        weights: Optional dictionary to weight the contributions of each component
    Returns:
        A quality score between 0.0 and 1.0.

    """
    if data.empty:
        return 0.0

    # 1. Completeness Score
    completeness_score = 1.0 - (data.isnull().sum().sum() / data.size)

    # 2. Consistency Score (high >= low)
    consistency_score = 1.0
    if 'high' in data.columns and 'low' in data.columns:
        consistency_subset = data[['high', 'low']].dropna()
        if not consistency_subset.empty:
            inconsistent_rows = (
                consistency_subset['high'] < consistency_subset['low']
            ).sum()
            consistency_score = 1.0 - (inconsistent_rows / len(consistency_subset))

    # 3. Volume Health Score
    volume_health_score = 1.0
    if 'volume' in data.columns:
        volume_series = data['volume'].dropna()
        if not volume_series.empty:
            zero_volume_ratio = (volume_series == 0).sum() / len(volume_series)
            # Penalize starting from a 10% threshold.
            if zero_volume_ratio > 0.1:
                # Scale penalty over a 20% range (10% to 30%)
                penalty_ratio = (zero_volume_ratio - 0.1) / 0.2
                volume_health_score = max(0.0, 1.0 - penalty_ratio)
            else:
                volume_health_score = 1.0

    # Final weighted score
    current_weights = weights or {
        "completeness": 0.4,
        "consistency": 0.4,
        "volume": 0.2,
    }
    final_score = (
        completeness_score * current_weights['completeness'] +
        consistency_score * current_weights['consistency'] +
        volume_health_score * current_weights['volume']
    )
    return max(0.0, min(1.0, final_score))

def check_data_gaps(
    data: pd.DataFrame, timestamp_column: str = 'timestamp'
) -> Dict[str, Any]:
    """Check for gaps in time series data.

    Args:
        data: DataFrame with time series data
        timestamp_column: Name of the timestamp column
    Returns:
        Dictionary with gap information

    """
    if timestamp_column not in data.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found.")

    df = data.copy()
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    df = df.sort_values(by=timestamp_column).set_index(timestamp_column)

    # Infer frequency
    inferred_freq = pd.infer_freq(df.index)
    if not inferred_freq:
        return {"error": "Could not infer frequency."}

    # Create expected time range
    expected_range = pd.date_range(
        start=df.index.min(), end=df.index.max(), freq=inferred_freq
    )
    missing_timestamps = expected_range.difference(df.index)

    gap_percentage = (
        (len(missing_timestamps) / len(expected_range)) * 100
        if expected_range.size > 0
        else 0
    )
    return {
        "inferred_frequency": inferred_freq,
        "total_gaps": len(missing_timestamps),
        "gap_percentage": gap_percentage,
        "missing_timestamps": missing_timestamps.to_list(),
    }
