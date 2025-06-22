"""Qlib storage wrapper for binary data format."""
import subprocess
from pathlib import Path
from typing import List
import pandas as pd

from src.types import Timeframe, DataSource
from src.market_data.contracts import QlibMarketData
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QlibStorage:
    """Wrapper for existing qlib dump_bin.py functionality."""
    
    def __init__(self):
        """Initialize qlib storage handler."""
        self.qlib_data_dir = Path("data/qlib_data")
        self.qlib_source_dir = Path("data/qlib_source")
    
    def store(
        self, 
        qlib_data: List[QlibMarketData], 
        symbol: str, 
        timeframe: Timeframe, 
        source: DataSource
    ) -> None:
        """
        Store data in qlib binary format using existing dump_bin.py.
        
        Args:
            qlib_data: List of QlibMarketData records
            symbol: Trading symbol
            timeframe: Data timeframe enum
            source: Data source enum
        """
        if not qlib_data:
            logger.warning("No qlib data to store")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([data.dict() for data in qlib_data])
        
        # Create source-specific directories
        source_dir = self.qlib_source_dir / source.value / timeframe.value
        source_dir.mkdir(parents=True, exist_ok=True)
        
        target_qlib_dir = self.qlib_data_dir / source.value
        target_qlib_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV file for qlib processing
        sanitized_symbol = symbol.replace("/", "")
        csv_file = source_dir / f"{sanitized_symbol}.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Saved qlib CSV: {csv_file}")
        
        # Use existing dump_bin.py to convert to binary format
        self._dump_to_binary(
            csv_path=source_dir,
            qlib_dir=target_qlib_dir,
            timeframe=timeframe,
            symbol=sanitized_symbol
        )
        
        logger.info(f"Stored qlib binary data: {target_qlib_dir}")
    
    def exists(self, symbol: str, timeframe: Timeframe, source: DataSource) -> bool:
        """
        Check if qlib binary data exists for the given parameters.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe enum
            source: Data source enum
            
        Returns:
            True if qlib binary data exists, False otherwise
        """
        sanitized_symbol = symbol.replace("/", "")
        source_qlib_dir = self.qlib_data_dir / source.value
        
        # Check for basic qlib structure
        calendars_exist = (source_qlib_dir / "calendars" / f"{timeframe.qlib_name}.txt").exists()
        instruments_exist = (source_qlib_dir / "instruments" / "all.txt").exists()
        features_exist = (source_qlib_dir / "features" / sanitized_symbol.lower()).exists()
        
        return calendars_exist and instruments_exist and features_exist
    
    def _dump_to_binary(
        self, 
        csv_path: Path, 
        qlib_dir: Path, 
        timeframe: Timeframe,
        symbol: str
    ) -> None:
        """
        Use internal qlib dump function to convert CSV to qlib binary format.
        
        Args:
            csv_path: Path to CSV files directory
            qlib_dir: Target qlib data directory
            timeframe: Data timeframe enum
            symbol: Symbol being processed
        """
        try:
            # Use the existing dump_bin.py script, now in market_data module
            dump_command = [
                "uv", "run", "python", "-m", "src.market_data.qlib.dump_bin", "dump_all",
                "--csv_path", str(csv_path),
                "--qlib_dir", str(qlib_dir),
                "--freq", timeframe.qlib_name,
                "--date_field_name", "date",
                "--symbol_field_name", "symbol",
                "--max_workers", "4",
                "--include_fields", "open,high,low,close,volume,factor"
            ]
            
            logger.info(f"Running qlib dump: {' '.join(dump_command)}")
            result = subprocess.run(
                dump_command,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("Qlib binary dump completed successfully")
            
            if result.stdout:
                logger.debug(f"Dump stdout: {result.stdout}")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Qlib dump failed: {e}")
            if e.stderr:
                logger.error(f"Dump stderr: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during qlib dump: {e}")
            raise 