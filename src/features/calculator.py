"""
Feature calculator for processing and generating features from market data.

Orchestrates the calculation of multiple technical indicators and
manages the feature generation pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .indicators.base import BaseIndicator
from .factory import feature_factory

# Use absolute import to avoid relative import issues
try:
    from utils.logger import get_logger
except ImportError:
    # Fallback for when utils module is not available
    import logging
    def get_logger(name):
        return logging.getLogger(name)


@dataclass
class FeatureConfig:
    """Configuration for feature calculation."""
    
    indicators: List[Dict[str, Any]]
    include_price_features: bool = True
    include_time_features: bool = True
    parallel_processing: bool = True
    max_workers: int = 4


class FeatureCalculator:
    """
    Main feature calculator for technical analysis.
    
    Handles multiple indicators and provides comprehensive
    feature engineering capabilities.
    """
    
    def __init__(self, config: FeatureConfig):
        """
        Initialize feature calculator.
        
        Args:
            config: Feature calculation configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.factory = feature_factory
        
        # Initialize indicators
        self.indicators = self._create_indicators()
        self.logger.info(f"Initialized {len(self.indicators)} indicators")
    
    def _create_indicators(self) -> List[BaseIndicator]:
        """
        Create indicator instances from configuration.
        
        Returns:
            List of configured indicators
        """
        indicators = []
        
        for indicator_config in self.config.indicators:
            try:
                indicator = self.factory.create_indicator_from_config(indicator_config)
                indicators.append(indicator)
                self.logger.debug(f"Created indicator: {indicator.config.name}")
            except Exception as e:
                self.logger.error(f"Failed to create indicator {indicator_config}: {e}")
        
        return indicators
    
    def calculate_features(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Calculate all features for the given data.
        
        Args:
            data: Input OHLCV DataFrame
            symbol: Optional symbol identifier
            
        Returns:
            DataFrame with original data plus calculated features
        """
        self.logger.info(f"Calculating features for {len(data)} data points")
        
        # Validate input data
        self._validate_input_data(data)
        
        # Start with original data
        result = data.copy()
        
        # Add basic price features
        if self.config.include_price_features:
            result = self._add_price_features(result)
        
        # Add time-based features
        if self.config.include_time_features:
            result = self._add_time_features(result)
        
        # Calculate technical indicators
        result = self._calculate_indicators(result)
        
        self.logger.info(f"Generated {len(result.columns)} total features")
        
        return result
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data format and requirements.
        
        Args:
            data: Input DataFrame to validate
            
        Raises:
            ValueError: If data validation fails
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(data) == 0:
            raise ValueError("Input data is empty")
        
        # Check for sufficient data for indicators
        max_required_periods = max([ind.get_required_periods() for ind in self.indicators], default=1)
        if len(data) < max_required_periods:
            self.logger.warning(
                f"Insufficient data for some indicators. "
                f"Available: {len(data)}, Required: {max_required_periods}"
            )
    
    def _add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic price-based features.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with price features added
        """
        result = data.copy()
        
        # Price changes
        result['price_change'] = data['close'].diff()
        result['price_change_pct'] = data['close'].pct_change()
        
        # High-Low spread
        result['hl_spread'] = data['high'] - data['low']
        result['hl_spread_pct'] = result['hl_spread'] / data['close']
        
        # Open-Close spread
        result['oc_spread'] = data['close'] - data['open']
        result['oc_spread_pct'] = result['oc_spread'] / data['open']
        
        # True Range
        prev_close = data['close'].shift(1)
        tr1 = data['high'] - data['low']
        tr2 = (data['high'] - prev_close).abs()
        tr3 = (data['low'] - prev_close).abs()
        result['true_range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Typical Price
        result['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
        
        # Price position within the range
        result['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        result['price_position'] = result['price_position'].fillna(0.5)  # Handle equal high/low
        
        self.logger.debug("Added basic price features")
        return result
    
    def _add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features.
        
        Args:
            data: Input DataFrame with datetime index
            
        Returns:
            DataFrame with time features added
        """
        result = data.copy()
        
        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.warning("Index is not datetime, skipping time features")
            return result
        
        # Basic time features
        result['hour'] = data.index.hour
        result['day_of_week'] = data.index.dayofweek
        result['day_of_month'] = data.index.day
        result['month'] = data.index.month
        
        # Cyclical encoding for time features
        result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
        result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
        result['day_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['day_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
        
        # Weekend flag
        result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)
        
        self.logger.debug("Added time-based features")
        return result
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with indicator features added
        """
        result = data.copy()
        
        for indicator in self.indicators:
            try:
                result = indicator.apply(result)
                self.logger.debug(f"Applied indicator: {indicator.config.name}")
            except Exception as e:
                self.logger.error(f"Failed to apply indicator {indicator.config.name}: {e}")
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names that will be generated.
        
        Returns:
            List of feature column names
        """
        feature_names = []
        
        # Base columns
        feature_names.extend(['open', 'high', 'low', 'close', 'volume'])
        
        # Price features
        if self.config.include_price_features:
            feature_names.extend([
                'price_change', 'price_change_pct', 'hl_spread', 'hl_spread_pct',
                'oc_spread', 'oc_spread_pct', 'true_range', 'typical_price', 'price_position'
            ])
        
        # Time features
        if self.config.include_time_features:
            feature_names.extend([
                'hour', 'day_of_week', 'day_of_month', 'month',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                'is_weekend'
            ])
        
        # Indicator features
        for indicator in self.indicators:
            feature_names.extend(indicator.get_feature_names())
        
        return feature_names
    
    def export_config(self) -> Dict[str, Any]:
        """
        Export current configuration for reproducibility.
        
        Returns:
            Configuration dictionary
        """
        return {
            'indicators': [ind.get_config_dict() for ind in self.indicators],
            'include_price_features': self.config.include_price_features,
            'include_time_features': self.config.include_time_features,
            'parallel_processing': self.config.parallel_processing,
            'max_workers': self.config.max_workers
        } 