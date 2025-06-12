"""
DataFrame and OHLCV data validation utilities.

This module provides validation functions for pandas DataFrames,
Series, OHLCV market data, and data quality assessment.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional

from src.exceptions import (
    ValidationError, MissingDataError, create_context
)


def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str], 
                              name: str = "DataFrame") -> pd.DataFrame:
    """
    Validate that a DataFrame has the required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        name: Parameter name for error messages
        
    Returns:
        The validated DataFrame
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"{name} must be a pandas DataFrame, got {type(df)}")
    
    if df.empty:
        raise ValidationError(f"{name} is empty")
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValidationError(f"Missing required columns in {name}: {missing_columns}")
    
    return df


def validate_series_numeric(series: pd.Series, allow_nan: bool = True, 
                           name: str = "Series") -> pd.Series:
    """
    Validate that a Series contains numeric data.
    
    Args:
        series: Series to validate
        allow_nan: Whether to allow NaN values
        name: Parameter name for error messages
        
    Returns:
        The validated Series
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(series, pd.Series):
        raise ValidationError(f"{name} must be a pandas Series, got {type(series)}")
    
    if not pd.api.types.is_numeric_dtype(series):
        raise ValidationError(f"{name} is not numeric")
    
    if not allow_nan and series.isnull().any():
        raise ValidationError(f"{name} contains NaN values")
    
    return series


def validate_ohlcv_data(data: pd.DataFrame, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Validate OHLCV data format and integrity.
    
    Args:
        data: DataFrame with OHLCV data
        required_columns: List of required column names (default: ['open', 'high', 'low', 'close', 'volume'])
        
    Returns:
        Validated DataFrame
        
    Raises:
        ValidationError: If validation fails
    """
    if required_columns is None:
        required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Check if input is DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValidationError(
            f"Expected pandas DataFrame, got {type(data)}",
            context=create_context(data_type=type(data))
        )
        
    # Check if DataFrame is empty
    if data.empty:
        raise ValidationError("DataFrame is empty")
        
    # Check required columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValidationError(
            f"Missing required OHLCV columns: {missing_columns}",
            context=create_context(missing_columns=missing_columns, required=required_columns)
        )
        
    # Check for numeric data types
    numeric_columns = [col for col in required_columns if col != 'symbol']
    for col in numeric_columns:
        if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
            raise ValidationError(
                f"Column {col} must be numeric",
                context=create_context(column=col, dtype=str(data[col].dtype))
            )
    
    # Check for negative prices
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        if col in data.columns and (data[col] <= 0).any():
            raise ValidationError(f"Column {col} contains non-positive values")
    
    # Check for negative volume
    if 'volume' in data.columns and (data['volume'] < 0).any():
        raise ValidationError("Volume cannot be negative")
        
    # Check for valid OHLC relationships (if all present)
    ohlc_cols = ['open', 'high', 'low', 'close']
    if all(col in data.columns for col in ohlc_cols):
        # High should be >= Open, Close, Low
        if not (data['high'] >= data[['open', 'close', 'low']].max(axis=1)).all():
            raise ValidationError("High prices must be >= Open, Close, and Low prices")
            
        # Low should be <= Open, Close, High
        if not (data['low'] <= data[['open', 'close', 'high']].min(axis=1)).all():
            raise ValidationError("Low prices must be <= Open, Close, and High prices")
        
    # Check for excessive NaN values
    nan_percentage = data.isnull().sum().sum() / (len(data) * len(data.columns))
    if nan_percentage > 0.1:  # More than 10% NaN
        raise ValidationError(
            f"Excessive missing data: {nan_percentage:.1%} of values are NaN",
            context=create_context(nan_percentage=nan_percentage)
        )
    
    return data


def validate_ohlcv_consistency(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate OHLCV data consistency (relationships between OHLC values).
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        Validated DataFrame
        
    Raises:
        ValidationError: If data is inconsistent
    """
    ohlc_cols = ['open', 'high', 'low', 'close']
    
    # Check if all OHLC columns are present
    missing_cols = [col for col in ohlc_cols if col not in data.columns]
    if missing_cols:
        raise ValidationError(f"Missing OHLC columns for consistency check: {missing_cols}")
    
    # Check for high < low condition
    if (data['high'] < data['low']).any():
        raise ValidationError("High price is less than low price")
    
    # Check if open/close are outside high-low range
    if (data['open'] > data['high']).any() or (data['open'] < data['low']).any():
        raise ValidationError("Open price is outside high-low range")
    
    if (data['close'] > data['high']).any() or (data['close'] < data['low']).any():
        raise ValidationError("Close price is outside high-low range")
    
    return data


def validate_data_completeness(data: pd.DataFrame, min_rows: int = 1,
                             max_gap_percentage: float = 5.0) -> pd.DataFrame:
    """
    Validate data completeness requirements.
    
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
            context=create_context(rows=len(data), min_rows=min_rows)
        )
    
    nan_count = data.isnull().sum().sum()
    total_values = len(data) * len(data.columns)
    nan_percentage = (nan_count / total_values * 100) if total_values > 0 else 0
    
    if nan_percentage > max_gap_percentage:
        raise ValidationError(
            f"Data contains {nan_percentage:.1f}% gaps, maximum {max_gap_percentage}% allowed",
            context=create_context(
                nan_percentage=nan_percentage,
                max_allowed=max_gap_percentage,
                nan_count=nan_count,
                total_values=total_values
            )
        )

    return data


def validate_data_quality(data: pd.DataFrame, min_quality_score: float = 0.8) -> pd.DataFrame:
    """
    Validate data quality and return the DataFrame if quality is acceptable.
    
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
            f"Data quality score {quality_score:.2f} below minimum {min_quality_score:.2f}",
            context=create_context(quality_score=quality_score, min_required=min_quality_score)
        )
    
    return data


def calculate_data_quality_score(data: pd.DataFrame) -> float:
    """
    Calculate a data quality score based on completeness and consistency.
    
    Args:
        data: DataFrame to analyze
        
    Returns:
        Quality score between 0.0 and 1.0
    """
    if data.empty:
        return 0.0
    
    # Calculate completeness score (1.0 - percentage of missing values)
    total_values = len(data) * len(data.columns)
    missing_values = data.isnull().sum().sum()
    completeness_score = 1.0 - (missing_values / total_values) if total_values > 0 else 0.0
    
    # For now, return just the completeness score
    # Could be extended with other quality metrics
    return max(0.0, min(1.0, completeness_score))


def check_data_gaps(data: pd.DataFrame, timestamp_column: str = 'timestamp') -> Dict[str, Any]:
    """
    Check for gaps in time series data.
    
    Args:
        data: DataFrame with time series data
        timestamp_column: Name of the timestamp column
        
    Returns:
        Dictionary with gap information
    """
    if data.empty:
        return {
            'total_gaps': 0,
            'gap_percentage': 0.0,
            'columns_with_gaps': []
        }
    
    # Count NaN values per column
    nan_counts = data.isnull().sum()
    total_gaps = nan_counts.sum()
    total_values = len(data) * len(data.columns)
    gap_percentage = (total_gaps / total_values * 100) if total_values > 0 else 0.0
    
    # Get columns with gaps
    columns_with_gaps = [col for col, count in nan_counts.items() if count > 0]
    
    return {
        'total_gaps': total_gaps,
        'gap_percentage': gap_percentage,
        'columns_with_gaps': columns_with_gaps
    } 