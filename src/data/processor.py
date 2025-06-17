"""Data processing pipeline for standardizing broker data.

This module handles data validation, cleaning, and standardization
to ensure consistent CSV format regardless of data source.
"""

from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

import pandas as pd

from ..utils.logger import get_logger
from .calendars.factory import calendar_factory


class DataProcessor:
    """Data processor for standardizing and validating trading data.

    Ensures all brokers produce identical CSV format:
    timestamp,open,high,low,close,volume
    2020-04-01T00:00:00Z,1.0856,1.0862,1.0854,1.0859,1250
    """

    REQUIRED_COLUMNS: ClassVar[list[str]] = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    TIMESTAMP_FORMAT: ClassVar[str] = "%Y-%m-%dT%H:%M:%SZ"

    def __init__(self, symbol: str, asset_class: str = "forex"):
        """Initialize data processor for a specific symbol.

        Args:
            symbol: Trading symbol (e.g., "EUR/USD")
            asset_class: Asset class for calendar selection

        """
        self.symbol = symbol
        self.asset_class = asset_class
        self.logger = get_logger(__name__)
        self.calendar = calendar_factory.create_calendar(asset_class)

    def standardize_dataframe(self, df: pd.DataFrame, broker_name: Optional[str] = None) -> pd.DataFrame:
        """Standardize DataFrame to consistent format.

        Args:
            df: Raw DataFrame from broker
            broker_name: Name of source broker for logging

        Returns:
            Standardized DataFrame with required columns and format

        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided for standardization")
            return self._create_empty_standard_df()

        self.logger.info(f"Standardizing {len(df)} records from {broker_name or 'unknown'}")

        # Create a copy to avoid modifying original
        standardized = df.copy()

        # Ensure we have required columns
        self._validate_required_columns(standardized)

        # Standardize timestamp format
        standardized = self._standardize_timestamps(standardized)

        # Validate and clean OHLCV data
        standardized = self._validate_ohlcv_data(standardized)

        # Sort by timestamp
        standardized = standardized.sort_values('timestamp')

        # Reset index
        standardized = standardized.reset_index(drop=True)

        # Ensure column order
        standardized = standardized[self.REQUIRED_COLUMNS]

        self.logger.info(f"Standardization complete. Final records: {len(standardized)}")

        return standardized

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """Validate that DataFrame has all required columns.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If required columns are missing

        """
        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"Required: {self.REQUIRED_COLUMNS}, "
                f"Found: {list(df.columns)}"
            )

    def _standardize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize timestamp column to ISO format with UTC timezone.

        Args:
            df: DataFrame with timestamp column

        Returns:
            DataFrame with standardized timestamps

        """
        if 'timestamp' not in df.columns:
            raise ValueError("No timestamp column found")

        # Convert to pandas datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Ensure UTC timezone
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')

        # Format as ISO string with Z suffix
        df['timestamp'] = df['timestamp'].dt.strftime(self.TIMESTAMP_FORMAT)

        return df

    def _validate_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean OHLCV data.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            Cleaned DataFrame

        """
        # Convert price columns to float
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert volume to int (default to 0 if missing)
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)

        # Validate OHLC relationships
        df = self._validate_ohlc_relationships(df)

        # Remove rows with NaN prices
        initial_count = len(df)
        df = df.dropna(subset=price_columns)
        removed_count = initial_count - len(df)

        if removed_count > 0:
            self.logger.warning(f"Removed {removed_count} rows with invalid price data")

        # Remove duplicate timestamps
        initial_count = len(df)
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
        removed_count = initial_count - len(df)

        if removed_count > 0:
            self.logger.warning(f"Removed {removed_count} duplicate timestamps")

        return df

    def _validate_ohlc_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLC relationships and fix minor inconsistencies.

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with validated OHLC relationships

        """
        # Check for invalid relationships
        invalid_high = df['high'] < df[['open', 'close']].max(axis=1)
        invalid_low = df['low'] > df[['open', 'close']].min(axis=1)

        if invalid_high.any():
            self.logger.warning(f"Found {invalid_high.sum()} bars with high < max(open,close)")
            # Fix by setting high to max of open/close
            df.loc[invalid_high, 'high'] = df.loc[invalid_high, ['open', 'close']].max(axis=1)

        if invalid_low.any():
            self.logger.warning(f"Found {invalid_low.sum()} bars with low > min(open,close)")
            # Fix by setting low to min of open/close
            df.loc[invalid_low, 'low'] = df.loc[invalid_low, ['open', 'close']].min(axis=1)

        return df

    def _create_empty_standard_df(self) -> pd.DataFrame:
        """Create empty DataFrame with standard columns.

        Returns:
            Empty standardized DataFrame

        """
        return pd.DataFrame(columns=self.REQUIRED_COLUMNS)

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data quality checks.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with quality metrics and issues

        """
        if df.empty:
            return {
                "total_records": 0,
                "quality_score": 0.0,
                "issues": ["Empty dataset"]
            }

        total_records = len(df)
        issues = []

        # Check for missing data
        missing_data = df.isnull().sum()
        if missing_data.any():
            issues.append(f"Missing data: {missing_data.to_dict()}")

        # Check for zero prices
        price_columns = ['open', 'high', 'low', 'close']
        zero_prices = (df[price_columns] <= 0).any(axis=1).sum()
        if zero_prices > 0:
            issues.append(f"Rows with zero/negative prices: {zero_prices}")

        # Check for suspicious price movements (>10% in one bar)
        if len(df) > 1:
            price_changes = df['close'].pct_change().abs()
            extreme_moves = (price_changes > 0.1).sum()
            if extreme_moves > 0:
                issues.append(f"Extreme price movements (>10%): {extreme_moves}")

        # Check data density (gaps in timestamps)
        if len(df) > 1:
            df_copy = df.copy()
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            df_copy = df_copy.sort_values('timestamp')

            # Calculate time differences
            time_diffs = df_copy['timestamp'].diff()
            median_diff = time_diffs.median()

            # Find gaps larger than 3x median interval
            large_gaps = (time_diffs > median_diff * 3).sum()
            if large_gaps > 0:
                issues.append(f"Large time gaps (>3x median): {large_gaps}")

        # Calculate quality score
        issues_count = len(issues)
        quality_score = max(0.0, 1.0 - (issues_count * 0.1))  # Each issue reduces score by 0.1

        return {
            "total_records": total_records,
            "quality_score": quality_score,
            "issues": issues,
            "date_range": {
                "start": df['timestamp'].min() if not df.empty else None,
                "end": df['timestamp'].max() if not df.empty else None
            }
        }

    def save_to_csv(
        self,
        df: pd.DataFrame,
        filepath: Path,
        metadata: Optional[Dict] = None
    ) -> None:
        """Save standardized DataFrame to CSV file.

        Args:
            df: Standardized DataFrame
            filepath: Path to save CSV file
            metadata: Optional metadata to log

        """
        if df.empty:
            self.logger.warning(f"Attempting to save empty DataFrame to {filepath}")
            return

        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save CSV (without index)
        df.to_csv(filepath, index=False)

        self.logger.info(f"Saved {len(df)} records to {filepath}")

        if metadata:
            self.logger.debug(f"Metadata: {metadata}")

    def filter_trading_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame to only include trading hours based on calendar.

        Args:
            df: DataFrame with timestamp column

        Returns:
            Filtered DataFrame with only trading hours

        """
        if df.empty:
            return df

        # Convert timestamps back to datetime for filtering
        df_copy = df.copy()
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])

        # Filter using calendar
        trading_times = self.calendar.filter_trading_times(df_copy['timestamp'].tolist())

        # Keep only rows with trading timestamps
        mask = df_copy['timestamp'].isin(trading_times)
        filtered_df = df[mask].copy()

        removed_count = len(df) - len(filtered_df)
        if removed_count > 0:
            self.logger.info(f"Filtered out {removed_count} non-trading hour records")

        return filtered_df
