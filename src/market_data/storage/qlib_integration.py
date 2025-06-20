"""Qlib integration for converting market data to qlib binary format."""

import os
import struct
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np

from src.market_data.contracts import MarketDataRequest, MarketDataResponse, QlibDataSpec
from src.market_data.exceptions import StorageError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QlibConverter:
    """Converts market data to qlib binary format."""
    
    def __init__(self, qlib_dir: str):
        """
        Initialize qlib converter.
        
        Args:
            qlib_dir: Directory for qlib data storage
        """
        self.qlib_dir = Path(qlib_dir)
        self.qlib_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized QlibConverter with qlib_dir: {self.qlib_dir}")
    
    def get_qlib_path(self, request: MarketDataRequest) -> str:
        """
        Get qlib binary file path for a request.
        
        Args:
            request: Market data request
            
        Returns:
            Path to qlib binary file
        """
        symbol_dir = self.qlib_dir / request.symbol.replace('/', '_')
        return str(symbol_dir / f"{request.timeframe.value}.bin")
    
    async def convert_data(self, response: MarketDataResponse) -> str:
        """
        Convert market data response to qlib binary format.
        
        Args:
            response: Market data response to convert
            
        Returns:
            Path to created qlib binary file
            
        Raises:
            StorageError: If conversion fails
        """
        try:
            request = response.request
            df = response.data
            
            # Create qlib data spec
            spec = QlibDataSpec(
                symbol=request.symbol,
                start_date=response.actual_start_date,
                end_date=response.actual_end_date,
                timeframe=request.timeframe,
                qlib_dir=str(self.qlib_dir)
            )
            
            # Ensure symbol directory exists
            symbol_dir = Path(spec.get_qlib_symbol_dir())
            symbol_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert DataFrame to qlib format
            qlib_data = self._prepare_qlib_data(df)
            
            # Write binary file
            qlib_file_path = spec.get_qlib_file_path()
            self._write_qlib_binary(qlib_data, qlib_file_path)
            
            logger.info(f"Converted {len(df)} bars to qlib format: {qlib_file_path}")
            return qlib_file_path
            
        except Exception as e:
            logger.error(f"Error converting to qlib: {e}")
            raise StorageError(f"Failed to convert to qlib: {e}") from e
    
    def _prepare_qlib_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for qlib binary format.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame prepared for qlib format
            
        Raises:
            StorageError: If data preparation fails
        """
        try:
            # Create a copy to avoid modifying original
            qlib_df = df.copy()
            
            # Ensure required columns exist
            required_columns = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
            if not required_columns.issubset(set(qlib_df.columns)):
                missing = required_columns - set(qlib_df.columns)
                raise StorageError(f"Missing required columns for qlib: {missing}")
            
            # Convert timestamp to qlib date format (YYYYMMDD)
            qlib_df['date'] = pd.to_datetime(qlib_df['timestamp']).dt.strftime('%Y%m%d').astype(int)
            
            # Rename columns to qlib format
            column_mapping = {
                'open': '$open',
                'high': '$high', 
                'low': '$low',
                'close': '$close',
                'volume': '$volume'
            }
            qlib_df = qlib_df.rename(columns=column_mapping)
            
            # Select and order columns for qlib
            qlib_columns = ['date', '$open', '$high', '$low', '$close', '$volume']
            qlib_df = qlib_df[qlib_columns]
            
            # Sort by date
            qlib_df = qlib_df.sort_values('date').reset_index(drop=True)
            
            # Ensure numeric types
            for col in ['$open', '$high', '$low', '$close', '$volume']:
                qlib_df[col] = pd.to_numeric(qlib_df[col], errors='coerce')
            
            # Remove any rows with NaN values
            qlib_df = qlib_df.dropna()
            
            logger.debug(f"Prepared {len(qlib_df)} rows for qlib format")
            return qlib_df
            
        except Exception as e:
            logger.error(f"Error preparing qlib data: {e}")
            raise StorageError(f"Failed to prepare qlib data: {e}") from e
    
    def _write_qlib_binary(self, df: pd.DataFrame, file_path: str) -> None:
        """
        Write DataFrame to qlib binary format.
        
        This implements the qlib binary format specification:
        - Header: number of records (4 bytes)
        - Records: each record contains date (4 bytes) + OHLCV data (5 * 4 bytes)
        
        Args:
            df: DataFrame with qlib-formatted data
            file_path: Output file path
            
        Raises:
            StorageError: If writing fails
        """
        try:
            with open(file_path, 'wb') as f:
                # Write header: number of records
                num_records = len(df)
                f.write(struct.pack('<I', num_records))
                
                # Write records
                for _, row in df.iterrows():
                    # Date (4 bytes, little-endian unsigned int)
                    f.write(struct.pack('<I', int(row['date'])))
                    
                    # OHLCV data (5 * 4 bytes, little-endian floats)
                    f.write(struct.pack('<f', float(row['$open'])))
                    f.write(struct.pack('<f', float(row['$high'])))
                    f.write(struct.pack('<f', float(row['$low'])))
                    f.write(struct.pack('<f', float(row['$close'])))
                    f.write(struct.pack('<f', float(row['$volume'])))
            
            file_size = os.path.getsize(file_path)
            logger.debug(f"Wrote qlib binary file: {file_path} ({file_size} bytes)")
            
        except Exception as e:
            logger.error(f"Error writing qlib binary: {e}")
            raise StorageError(f"Failed to write qlib binary: {e}") from e
    
    def read_qlib_binary(self, file_path: str) -> pd.DataFrame:
        """
        Read qlib binary file back to DataFrame.
        
        Args:
            file_path: Path to qlib binary file
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            StorageError: If reading fails
        """
        try:
            if not os.path.exists(file_path):
                raise StorageError(f"Qlib binary file not found: {file_path}")
            
            records = []
            
            with open(file_path, 'rb') as f:
                # Read header: number of records
                num_records_bytes = f.read(4)
                if len(num_records_bytes) < 4:
                    raise StorageError("Invalid qlib binary file: missing header")
                
                num_records = struct.unpack('<I', num_records_bytes)[0]
                
                # Read records
                for _ in range(num_records):
                    # Read date (4 bytes)
                    date_bytes = f.read(4)
                    if len(date_bytes) < 4:
                        break
                    date = struct.unpack('<I', date_bytes)[0]
                    
                    # Read OHLCV (5 * 4 bytes)
                    ohlcv_bytes = f.read(20)  # 5 * 4 bytes
                    if len(ohlcv_bytes) < 20:
                        break
                    
                    ohlcv = struct.unpack('<fffff', ohlcv_bytes)
                    
                    records.append({
                        'date': date,
                        'open': ohlcv[0],
                        'high': ohlcv[1],
                        'low': ohlcv[2],
                        'close': ohlcv[3],
                        'volume': ohlcv[4]
                    })
            
            # Convert to DataFrame
            df = pd.DataFrame(records)
            
            if not df.empty:
                # Convert date back to timestamp
                df['timestamp'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
                df = df.drop('date', axis=1)
            
            logger.debug(f"Read {len(df)} records from qlib binary: {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error reading qlib binary: {e}")
            raise StorageError(f"Failed to read qlib binary: {e}") from e
    
    def get_qlib_info(self, symbol: str) -> dict:
        """
        Get information about qlib data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with qlib data information
        """
        try:
            symbol_dir = self.qlib_dir / symbol.replace('/', '_')
            
            if not symbol_dir.exists():
                return {"exists": False, "symbol": symbol}
            
            # Find all timeframe files
            timeframes = []
            total_size = 0
            
            for file_path in symbol_dir.glob("*.bin"):
                timeframe = file_path.stem
                file_size = file_path.stat().st_size
                total_size += file_size
                
                # Try to read basic info
                try:
                    df = self.read_qlib_binary(str(file_path))
                    bars_count = len(df)
                    date_range = (df['timestamp'].min(), df['timestamp'].max()) if not df.empty else (None, None)
                except Exception:
                    bars_count = 0
                    date_range = (None, None)
                
                timeframes.append({
                    "timeframe": timeframe,
                    "file_size_bytes": file_size,
                    "bars_count": bars_count,
                    "start_date": date_range[0].isoformat() if date_range[0] else None,
                    "end_date": date_range[1].isoformat() if date_range[1] else None
                })
            
            return {
                "exists": True,
                "symbol": symbol,
                "symbol_dir": str(symbol_dir),
                "total_size_bytes": total_size,
                "timeframes": timeframes
            }
            
        except Exception as e:
            logger.error(f"Error getting qlib info: {e}")
            return {"exists": False, "symbol": symbol, "error": str(e)} 