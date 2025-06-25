"""Data processing pipeline for training experiments."""

import traceback
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime, timezone, timedelta
import pandas as pd

from src.market_data import prepare_training_data
from src.market_data.features import build_features
from src.types import Timeframe, DataSource
from src.market_data.exceptions import DataSourceError, DataRangeError, StorageError
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def process_training_data(
    experiment_id: str,
    config: Dict[str, Any],
    status_callback=None
) -> pd.DataFrame:
    """Process market data through the unified market data pipeline.
    
    This function uses the new unified market data API to:
    1. Download and cache market data from broker
    2. Convert to qlib binary format 
    3. Generate technical indicator features
    
    Args:
        experiment_id: Experiment identifier
        config: Training configuration
        status_callback: Optional callback for status updates
        
    Returns:
        features_df
        
    Raises:
        DataSourceError: If data processing fails
        DataRangeError: If data range is invalid
        StorageError: If storage operations fail
    """
    symbol = config["symbol"]
    timeframe_str = config["timeframe"]
    
    try:
        # Convert timeframe string to enum
        timeframe = Timeframe.from_standard(timeframe_str)
    except ValueError as e:
        # Propagate as ValueError so callers/tests can rely on standard exception type
        raise ValueError(f"Invalid timeframe: {timeframe_str}. Valid options: {Timeframe.get_display_list()}") from e
    
    logger.info(f"Processing training data for {symbol} {timeframe.value}")
    
    try:
        # Step 1: Prepare training data using unified API
        if status_callback:
            await status_callback("Downloading market data...")
        
        # Calculate date range for training (2 years of data)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=730)  # 2 years
        
        logger.info(f"Fetching {symbol} data from {start_date.date()} to {end_date.date()}")
        
        # Use unified market data API
        df = await prepare_training_data(
            symbol=symbol,
            source=DataSource.FOREX_COM,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            raise DataSourceError(f"No market data retrieved for {symbol}")
        
        logger.info(f"Retrieved {len(df)} bars of market data for {symbol}")
        
        # Step 2: Convert to qlib binary format
        if status_callback:
            await status_callback("Converting to qlib binary format...")
        
        # Prepare data for qlib
        from src.market_data.download import prepare_for_qlib
        qlib_df = prepare_for_qlib(df)
        
        # Save CSV for qlib processing
        sanitized_symbol = symbol.replace("/", "")
        qlib_source_dir = Path(f"data/qlib_source/{timeframe_str}")
        qlib_source_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = qlib_source_dir / f"{sanitized_symbol}.csv"
        qlib_df.to_csv(csv_path)
        logger.info(f"Saved qlib CSV data: {csv_path}")
        
        # Convert CSV to qlib binary format
        from src.market_data.qlib.dump_bin import DumpDataAll
        qlib_data_dir = Path("data/qlib_data")
        qlib_data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Converting CSV data to qlib binary format in {qlib_data_dir}")
        
        dumper = DumpDataAll(
            csv_path=str(qlib_source_dir),
            qlib_dir=str(qlib_data_dir),
            freq=timeframe.qlib_name,
            date_field_name="date",
            symbol_field_name="symbol",
            max_workers=4,
            include_fields="open,high,low,close,volume,factor"
        )
        
        dumper.dump()
        
        # Verify qlib binary data was created
        if not qlib_data_dir.exists() or not any(qlib_data_dir.iterdir()):
            raise DataSourceError("Qlib binary data not generated properly")
        
        logger.info(f"Qlib binary data successfully created in: {qlib_data_dir}")
        
        # Step 3: Build features using qlib
        if status_callback:
            await status_callback("Building training features...")
        
        build_features(symbol=symbol, timeframe=timeframe_str)
        
        # Verify features file exists
        features_path = Path("data/processed/features") / f"{sanitized_symbol}_{timeframe_str}_features.csv"
        
        if not features_path.exists():
            raise DataSourceError(f"Features file not generated: {features_path}")
        
        # Load and validate features
        features_df = pd.read_csv(features_path)
        
        if features_df.empty:
            raise DataSourceError(f"Generated features file is empty: {features_path}")
        
        logger.info(f"Successfully generated {len(features_df)} rows of features with {len(features_df.columns)} columns")
        
        return features_df
        
    except (DataSourceError, DataRangeError, StorageError) as e:
        error_msg = f"Failed to process training data for {symbol} {timeframe_str}: {e}"
        logger.error(error_msg)
        raise
    except Exception as e:
        error_msg = f"Unexpected error processing training data for {symbol} {timeframe_str}: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise DataSourceError(error_msg) from e 