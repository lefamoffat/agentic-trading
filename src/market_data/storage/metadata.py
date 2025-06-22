"""Data span tracking for intelligent caching."""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from src.types import Timeframe, DataSource
from src.market_data.contracts import DateRange, DataSpanInfo
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DataSpanTracker:
    """Track what data ranges we have stored for intelligent caching."""
    
    def __init__(self):
        """Initialize metadata tracker."""
        self.metadata_file = Path("data/metadata/data_spans.json")
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
    
    def record_exact_range(
        self, 
        symbol: str, 
        timeframe: Timeframe, 
        source: DataSource,
        date_range: DateRange,
        total_bars: int
    ) -> None:
        """
        Record what exact data range we have stored.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe enum
            source: Data source enum
            date_range: Date range that was stored
            total_bars: Number of bars stored
        """
        metadata = self._load_metadata()
        
        key = self._get_metadata_key(symbol, timeframe, source)
        
        span_info = DataSpanInfo(
            symbol=symbol,
            start_date=date_range.start_date,
            end_date=date_range.end_date,
            total_bars=total_bars,
            stored_at=datetime.utcnow(),
            source=source.value,
            timeframe=timeframe.value
        )
        
        metadata[key] = span_info.dict()
        self._save_metadata(metadata)
        
        logger.info(
            f"Recorded data span: {symbol} {timeframe.value} "
            f"from {date_range.start_date} to {date_range.end_date} ({total_bars} bars)"
        )
    
    def has_exact_range(
        self, 
        symbol: str, 
        timeframe: Timeframe, 
        source: DataSource,
        requested_range: DateRange
    ) -> bool:
        """
        Check if we have the exact requested data range stored.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe enum
            source: Data source enum
            requested_range: Date range being requested
            
        Returns:
            True if we have exact range stored, False otherwise
        """
        metadata = self._load_metadata()
        key = self._get_metadata_key(symbol, timeframe, source)
        
        if key not in metadata:
            return False
        
        stored_info = metadata[key]
        stored_start = datetime.fromisoformat(stored_info["start_date"].replace('Z', '+00:00'))
        stored_end = datetime.fromisoformat(stored_info["end_date"].replace('Z', '+00:00'))
        
        # Check if stored range covers requested range
        return (
            stored_start <= requested_range.start_date and
            stored_end >= requested_range.end_date
        )
    
    def get_data_info(
        self, 
        symbol: str, 
        timeframe: Timeframe, 
        source: DataSource
    ) -> Optional[DataSpanInfo]:
        """
        Get information about stored data.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe enum
            source: Data source enum
            
        Returns:
            DataSpanInfo if data exists, None otherwise
        """
        metadata = self._load_metadata()
        key = self._get_metadata_key(symbol, timeframe, source)
        
        if key not in metadata:
            return None
        
        return DataSpanInfo(**metadata[key])
    
    def _get_metadata_key(self, symbol: str, timeframe: Timeframe, source: DataSource) -> str:
        """Generate unique key for metadata storage."""
        sanitized_symbol = symbol.replace("/", "")
        return f"{sanitized_symbol}_{timeframe.value}_{source.value}"
    
    def _load_metadata(self) -> Dict:
        """Load metadata from JSON file."""
        if not self.metadata_file.exists():
            return {}
        
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load metadata: {e}, starting fresh")
            return {}
    
    def _save_metadata(self, metadata: Dict) -> None:
        """Save metadata to JSON file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Failed to save metadata: {e}")
            raise 