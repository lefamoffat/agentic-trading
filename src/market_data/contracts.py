"""Data contracts for market data using Pydantic for type safety.

These contracts define the structure and validation rules for market data
requests and responses, ensuring data integrity throughout the pipeline.
"""

from datetime import datetime, timezone
from typing import Optional, List
import pandas as pd
from pydantic import BaseModel, validator, Field, field_validator, ConfigDict

from src.types import Timeframe, DataSource

class DateRange(BaseModel):
    """Date range specification for historical data requests."""
    
    model_config = ConfigDict(extra='forbid')
    
    start_date: datetime
    end_date: datetime
    
    @field_validator('start_date', 'end_date')
    @classmethod
    def ensure_utc(cls, v):
        """Ensure all timestamps are UTC."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)
    
    @field_validator('end_date')
    @classmethod
    def end_after_start(cls, v, info):
        """Ensure end_date is after start_date."""
        if info.data.get('start_date') and v <= info.data['start_date']:
            raise ValueError('end_date must be after start_date')
        return v

class QlibMarketData(BaseModel):
    """Market data formatted for Qlib storage."""
    
    date: datetime  # Qlib uses 'date' column name
    open: float
    high: float
    low: float
    close: float
    volume: int
    factor: float = Field(default=1.0, description="Stock split adjustment factor")
    
    @validator('date')
    def ensure_timezone_naive(cls, v):
        """Qlib requires timezone-naive timestamps."""
        if v.tzinfo is not None:
            return v.replace(tzinfo=None)
        return v

class DataSpanInfo(BaseModel):
    """Information about stored data span."""
    
    symbol: str
    start_date: datetime
    end_date: datetime
    total_bars: int
    stored_at: datetime
    source: str  # DataSource enum value
    timeframe: str  # Timeframe enum value
    
    @validator('stored_at', 'start_date', 'end_date')
    def ensure_utc(cls, v):
        """Ensure all timestamps are UTC."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

class DataRequestParams(BaseModel):
    """Parameters for market data requests."""
    symbol: str = Field(..., description="Trading symbol (e.g., 'EUR/USD')")
    timeframe: Timeframe = Field(..., description="Data timeframe enum")
    source: DataSource = Field(..., description="Data source enum")
    date_range: DateRange = Field(..., description="Date range to fetch")
    
    @validator('symbol')
    def symbol_not_empty(cls, v):
        """Ensure symbol is not empty."""
        if not v or not v.strip():
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()

class MarketDataRequest(BaseModel):
    """Request for historical market data."""
    
    model_config = ConfigDict(extra='forbid')  # Reject extra fields in v2
    
    symbol: str = Field(..., description="Trading symbol (e.g., 'EUR/USD')")
    source: DataSource = Field(..., description="Data source to use")
    timeframe: Timeframe = Field(..., description="Data timeframe")
    start_date: datetime = Field(..., description="Start date (UTC)")
    end_date: datetime = Field(..., description="End date (UTC)")
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v):
        """Validate symbol format."""
        if not v or not v.strip():
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()
    
    @field_validator('end_date')
    @classmethod
    def validate_date_range(cls, v, info):
        """Validate that end_date is after start_date."""
        if info.data.get('start_date') and v <= info.data['start_date']:
            raise ValueError("end_date must be after start_date")
        return v
    
    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_timezone(cls, v):
        """Ensure dates are timezone-aware (UTC)."""
        if v.tzinfo is None:
            raise ValueError("Dates must be timezone-aware (UTC)")
        return v
    
    def get_cache_key(self) -> str:
        """Generate unique cache key for this request."""
        return (
            f"{self.symbol}_{self.source.value}_{self.timeframe.value}_"
            f"{self.start_date.isoformat()}_{self.end_date.isoformat()}"
        )

class MarketDataResponse(BaseModel):
    """Response containing market data and metadata."""
    
    request: MarketDataRequest = Field(..., description="Original request")
    data: pd.DataFrame = Field(..., description="OHLCV data")
    bars_count: int = Field(..., ge=0, description="Number of bars returned")
    actual_start_date: datetime = Field(..., description="Actual start date of data")
    actual_end_date: datetime = Field(..., description="Actual end date of data")
    cached: bool = Field(default=False, description="Whether data came from cache")
    cache_timestamp: Optional[datetime] = Field(None, description="When data was cached")
    
    @validator('data')
    def validate_dataframe_structure(cls, v):
        """Validate DataFrame has required OHLCV columns."""
        required_columns = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        if not required_columns.issubset(set(v.columns)):
            missing = required_columns - set(v.columns)
            raise ValueError(f"DataFrame missing required columns: {missing}")
        return v
    
    @validator('bars_count')
    def validate_bars_count_matches_data(cls, v, values):
        """Validate bars_count matches actual DataFrame length."""
        if 'data' in values and len(values['data']) != v:
            raise ValueError(f"bars_count ({v}) doesn't match DataFrame length ({len(values['data'])})")
        return v
    
    @validator('actual_start_date', 'actual_end_date')
    def validate_actual_dates_timezone(cls, v):
        """Ensure actual dates are timezone-aware."""
        if v.tzinfo is None:
            raise ValueError("Actual dates must be timezone-aware")
        return v
    
    def get_date_range_coverage(self) -> float:
        """
        Calculate how much of the requested date range is covered.
        
        Returns:
            Coverage ratio (0.0 to 1.0)
        """
        requested_duration = (self.request.end_date - self.request.start_date).total_seconds()
        actual_duration = (self.actual_end_date - self.actual_start_date).total_seconds()
        
        if requested_duration <= 0:
            return 0.0
        
        return min(1.0, actual_duration / requested_duration)
    
    def is_complete_coverage(self, tolerance_hours: float = 24.0) -> bool:
        """
        Check if the response provides complete coverage of the requested range.
        
        Args:
            tolerance_hours: Acceptable gap in hours at start/end
            
        Returns:
            True if coverage is complete within tolerance
        """
        tolerance_delta = pd.Timedelta(hours=tolerance_hours)
        
        start_gap = abs(self.actual_start_date - self.request.start_date)
        end_gap = abs(self.actual_end_date - self.request.end_date)
        
        return start_gap <= tolerance_delta and end_gap <= tolerance_delta
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True  # Allow pandas DataFrame

class CacheMetadata(BaseModel):
    """Metadata for cached market data."""
    
    request_hash: str = Field(..., description="Hash of the original request")
    symbol: str = Field(..., description="Trading symbol")
    source: DataSource = Field(..., description="Data source")
    timeframe: Timeframe = Field(..., description="Data timeframe")
    start_date: datetime = Field(..., description="Data start date")
    end_date: datetime = Field(..., description="Data end date")
    bars_count: int = Field(..., ge=0, description="Number of bars")
    cached_at: datetime = Field(..., description="When data was cached")
    file_path: str = Field(..., description="Path to cached data file")
    file_size_bytes: int = Field(..., ge=0, description="File size in bytes")
    
    @validator('cached_at')
    def validate_cached_at_timezone(cls, v):
        """Ensure cached_at is timezone-aware."""
        if v.tzinfo is None:
            raise ValueError("cached_at must be timezone-aware")
        return v
    
    def is_expired(self, max_age_hours: float = 24.0) -> bool:
        """
        Check if cached data has expired.
        
        Args:
            max_age_hours: Maximum age in hours before considering expired
            
        Returns:
            True if data is expired
        """
        age = datetime.now(self.cached_at.tzinfo) - self.cached_at
        max_age = pd.Timedelta(hours=max_age_hours)
        return age > max_age
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True  # Allow pandas DataFrame

class QlibDataSpec(BaseModel):
    """Specification for qlib data conversion."""
    
    symbol: str = Field(..., description="Symbol for qlib")
    start_date: datetime = Field(..., description="Start date")
    end_date: datetime = Field(..., description="End date")
    timeframe: Timeframe = Field(..., description="Timeframe")
    qlib_dir: str = Field(..., description="Qlib data directory path")
    
    def get_qlib_symbol_dir(self) -> str:
        """Get qlib symbol directory path."""
        return f"{self.qlib_dir}/{self.symbol.replace('/', '_')}"
    
    def get_qlib_file_path(self) -> str:
        """Get qlib binary file path."""
        return f"{self.get_qlib_symbol_dir()}/{self.timeframe.value}.bin"
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True  # Allow pandas DataFrame

 