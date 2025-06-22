"""Cache manager for market data with intelligent caching and metadata tracking."""

import json
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd

from src.market_data.contracts import MarketDataRequest, MarketDataResponse, CacheMetadata
from src.market_data.exceptions import StorageError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CacheManager:
    """Manages caching of market data with metadata tracking."""
    
    def __init__(self, cache_dir: str):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file for cache index
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self._metadata_cache: Dict[str, CacheMetadata] = {}
        
        # Load existing metadata
        self._load_metadata()
        
        logger.info(f"Initialized CacheManager with cache_dir: {self.cache_dir}")
    
    def _load_metadata(self) -> None:
        """Load cache metadata from disk."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                
                # Convert to CacheMetadata objects
                for key, data in metadata_dict.items():
                    try:
                        # Convert datetime strings back to datetime objects
                        for date_field in ['start_date', 'end_date', 'cached_at']:
                            if date_field in data:
                                data[date_field] = datetime.fromisoformat(data[date_field])
                        
                        # Convert enum strings back to enums
                        if 'source' in data and isinstance(data['source'], str):
                            from src.types import DataSource
                            data['source'] = DataSource(data['source'])
                        if 'timeframe' in data and isinstance(data['timeframe'], str):
                            from src.types import Timeframe
                            data['timeframe'] = Timeframe(data['timeframe'])
                        
                        self._metadata_cache[key] = CacheMetadata(**data)
                    except Exception as e:
                        logger.warning(f"Failed to load metadata for {key}: {e}")
                
                logger.info(f"Loaded {len(self._metadata_cache)} cache entries")
            else:
                logger.info("No existing cache metadata found")
                
        except Exception as e:
            logger.error(f"Error loading cache metadata: {e}")
            self._metadata_cache = {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            # Convert to serializable format
            metadata_dict = {}
            for key, metadata in self._metadata_cache.items():
                data = metadata.dict()
                # Convert datetime objects to strings
                for date_field in ['start_date', 'end_date', 'cached_at']:
                    if date_field in data and data[date_field]:
                        data[date_field] = data[date_field].isoformat()
                
                # Convert enums to strings
                if 'source' in data and hasattr(data['source'], 'value'):
                    data['source'] = data['source'].value
                if 'timeframe' in data and hasattr(data['timeframe'], 'value'):
                    data['timeframe'] = data['timeframe'].value
                    
                metadata_dict[key] = data
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")
    
    def _get_cache_key(self, request: MarketDataRequest) -> str:
        """Generate cache key for request."""
        return request.get_cache_key()
    
    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Get cache file path for a cache key."""
        # Use hash to avoid filesystem issues with long names
        key_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.parquet"
    
    async def get_cached_data(self, request: MarketDataRequest) -> Optional[pd.DataFrame]:
        """
        Get cached data for a request if available and valid.
        
        Args:
            request: Market data request
            
        Returns:
            Cached DataFrame or None if not available
        """
        try:
            # Search through all cached entries to find one that can satisfy this request
            for cache_key, metadata in self._metadata_cache.items():
                # Check if this cached entry matches the symbol, source, and timeframe
                if (metadata.symbol != request.symbol or 
                    metadata.source != request.source or 
                    metadata.timeframe != request.timeframe):
                    continue
                
                # Check if cache file exists
                cache_file = Path(metadata.file_path)
                if not cache_file.exists():
                    logger.warning(f"Cache file missing: {cache_file}")
                    # Note: We don't remove stale metadata here since we're iterating
                    continue
                
                # Check if cache is expired (default 24 hours)
                if metadata.is_expired():
                    logger.info(f"Cache expired for {request.symbol}")
                    continue
                
                # Check if cached data range covers the requested range
                if (request.start_date >= metadata.start_date and 
                    request.end_date <= metadata.end_date):
                    
                    # Load cached data
                    df = pd.read_parquet(cache_file)
                    
                    # Filter to exact requested range
                    if 'timestamp' in df.columns:
                        df = df[
                            (df['timestamp'] >= request.start_date) & 
                            (df['timestamp'] <= request.end_date)
                        ].copy()
                    
                    logger.info(f"Cache hit for {request.symbol}: {len(df)} bars")
                    return df
            
            # No suitable cached data found
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached data: {e}")
            return None
    
    async def cache_data(self, response: MarketDataResponse) -> None:
        """
        Cache market data response.
        
        Args:
            response: Market data response to cache
            
        Raises:
            StorageError: If caching fails
        """
        try:
            cache_key = self._get_cache_key(response.request)
            cache_file = self._get_cache_file_path(cache_key)
            
            # Save data as parquet for efficient storage
            response.data.to_parquet(cache_file, index=False)
            
            # Create metadata
            file_size = cache_file.stat().st_size
            metadata = CacheMetadata(
                request_hash=cache_key,
                symbol=response.request.symbol,
                source=response.request.source,
                timeframe=response.request.timeframe,
                start_date=response.actual_start_date,
                end_date=response.actual_end_date,
                bars_count=response.bars_count,
                cached_at=datetime.now(timezone.utc),
                file_path=str(cache_file),
                file_size_bytes=file_size
            )
            
            # Update metadata cache
            self._metadata_cache[cache_key] = metadata
            self._save_metadata()
            
            logger.info(f"Cached {response.bars_count} bars for {response.request.symbol} "
                       f"({file_size / 1024:.1f} KB)")
            
        except Exception as e:
            logger.error(f"Error caching data: {e}")
            raise StorageError(f"Failed to cache data: {e}") from e
    
    def cleanup_expired(self, max_age_hours: float = 24.0) -> int:
        """
        Clean up expired cache entries.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of entries cleaned up
        """
        try:
            expired_keys = []
            
            for cache_key, metadata in self._metadata_cache.items():
                if metadata.is_expired(max_age_hours):
                    expired_keys.append(cache_key)
                    
                    # Remove cache file
                    cache_file = Path(metadata.file_path)
                    if cache_file.exists():
                        cache_file.unlink()
                        logger.debug(f"Removed expired cache file: {cache_file}")
            
            # Remove from metadata
            for key in expired_keys:
                del self._metadata_cache[key]
            
            if expired_keys:
                self._save_metadata()
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
            
            return len(expired_keys)
            
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
            return 0
    
    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            total_entries = len(self._metadata_cache)
            total_size = sum(metadata.file_size_bytes for metadata in self._metadata_cache.values())
            
            # Group by symbol and source
            symbols = set(metadata.symbol for metadata in self._metadata_cache.values())
            sources = set(metadata.source for metadata in self._metadata_cache.values())
            
            # Calculate age distribution
            now = datetime.now(timezone.utc)
            age_hours = [(now - metadata.cached_at).total_seconds() / 3600 
                        for metadata in self._metadata_cache.values()]
            
            return {
                "total_entries": total_entries,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "unique_symbols": len(symbols),
                "unique_sources": len(sources),
                "avg_age_hours": round(sum(age_hours) / len(age_hours), 2) if age_hours else 0,
                "max_age_hours": round(max(age_hours), 2) if age_hours else 0,
                "symbols": sorted(list(symbols)),
                "sources": sorted([source.value for source in sources])
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}
    
    def clear_cache(self) -> int:
        """
        Clear all cache data.
        
        Returns:
            Number of entries cleared
        """
        try:
            count = len(self._metadata_cache)
            
            # Remove all cache files
            for metadata in self._metadata_cache.values():
                cache_file = Path(metadata.file_path)
                if cache_file.exists():
                    cache_file.unlink()
            
            # Clear metadata
            self._metadata_cache.clear()
            self._save_metadata()
            
            logger.info(f"Cleared {count} cache entries")
            return count
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0 