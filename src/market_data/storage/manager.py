"""Storage manager for coordinating caching and qlib integration."""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import pandas as pd

from src.market_data.contracts import MarketDataRequest, MarketDataResponse, CacheMetadata
from src.market_data.storage.cache import CacheManager
from src.market_data.storage.qlib_integration import QlibConverter
from src.market_data.exceptions import StorageError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StorageManager:
    """Manages data storage, caching, and qlib integration."""
    
    def __init__(self, storage_dir: str = "data/cache"):
        """
        Initialize storage manager.
        
        Args:
            storage_dir: Base directory for market data storage
        """
        self.storage_dir = Path(storage_dir)
        self.cache_dir = self.storage_dir  # Use storage_dir directly for cache
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache manager
        self.cache_manager = CacheManager(str(self.cache_dir))
        
        # Note: qlib conversion is handled by separate dump_bin script
        self.qlib_converter = None
        
        logger.info(f"Initialized StorageManager with storage_dir: {self.storage_dir}")
    
    async def get_cached_data(self, request: MarketDataRequest) -> Optional[pd.DataFrame]:
        """
        Get cached data for a request if available and valid.
        
        Args:
            request: Market data request
            
        Returns:
            Cached DataFrame or None if not available
            
        Raises:
            StorageError: If cache operations fail
        """
        try:
            return await self.cache_manager.get_cached_data(request)
        except Exception as e:
            logger.error(f"Error getting cached data: {e}")
            raise StorageError(f"Failed to get cached data: {e}") from e
    
    async def cache_data(self, response: MarketDataResponse) -> None:
        """
        Cache market data response for future use.
        
        Args:
            response: Market data response to cache
            
        Raises:
            StorageError: If caching fails
        """
        try:
            await self.cache_manager.cache_data(response)
            logger.info(f"Cached {response.bars_count} bars for {response.request.symbol}")
        except Exception as e:
            logger.error(f"Error caching data: {e}")
            raise StorageError(f"Failed to cache data: {e}") from e
    
    async def convert_to_qlib(self, response: MarketDataResponse) -> str:
        """
        Convert market data to qlib binary format.
        
        Note: This functionality is now handled by the separate dump_bin script.
        
        Args:
            response: Market data response to convert
            
        Returns:
            Path to qlib binary file
            
        Raises:
            StorageError: If conversion fails
        """
        raise StorageError("Qlib conversion is handled by the separate dump_bin script. Use the CLI prepare-data command instead.")
    
    async def prepare_for_qlib(self, request: MarketDataRequest) -> str:
        """
        Ensure data is available in qlib format, fetching if necessary.
        
        Note: This functionality is now handled by the separate dump_bin script.
        
        Args:
            request: Market data request
            
        Returns:
            Path to qlib binary file
            
        Raises:
            StorageError: If preparation fails
        """
        raise StorageError("Qlib preparation is handled by the separate dump_bin script. Use the CLI prepare-data command instead.")
    
    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            return self.cache_manager.get_cache_stats()
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}
    
    def cleanup_expired_cache(self, max_age_hours: float = 24.0) -> int:
        """
        Clean up expired cache entries.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of entries cleaned up
        """
        try:
            return self.cache_manager.cleanup_expired(max_age_hours)
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
            return 0
    
    def get_storage_info(self) -> dict:
        """
        Get storage information and statistics.
        
        Returns:
            Dictionary with storage information
        """
        try:
            cache_stats = self.get_cache_stats()
            
            # Calculate directory sizes
            cache_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
            
            return {
                "storage_dir": str(self.storage_dir),
                "cache_dir": str(self.cache_dir),
                "cache_size_mb": round(cache_size / (1024 * 1024), 2),
                "cache_stats": cache_stats
            }
        except Exception as e:
            logger.error(f"Error getting storage info: {e}")
            return {"error": str(e)}


# Global storage manager instance
storage_manager = StorageManager() 