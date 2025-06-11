"""
Base indicator interface for technical indicators.

This provides a consistent API for all technical indicators,
making them easily composable and configurable.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

# Use absolute import to avoid relative import issues
try:
    from utils.logger import get_logger
except ImportError:
    # Fallback for when utils module is not available
    import logging
    def get_logger(name):
        return logging.getLogger(name)


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""
    
    name: str
    indicator_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_columns: List[str] = field(default_factory=lambda: ['close'])
    output_columns: List[str] = field(default_factory=list)
    min_periods: int = 1
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.output_columns:
            self.output_columns = [self.name]


class BaseIndicator(ABC):
    """
    Abstract base class for all technical indicators.
    
    Provides consistent interface for calculation, validation,
    and configuration management.
    """
    
    def __init__(self, config: IndicatorConfig):
        """
        Initialize indicator with configuration.
        
        Args:
            config: Indicator configuration
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.{config.name}")
        self._validate_config()
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicator values.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with calculated indicator values
        """
        pass
    
    @abstractmethod
    def get_required_periods(self) -> int:
        """
        Get minimum number of periods required for calculation.
        
        Returns:
            Minimum periods needed
        """
        pass
    
    def validate_input(self, data: pd.DataFrame) -> None:
        """
        Validate input data has required columns and sufficient periods.
        
        Args:
            data: Input DataFrame to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Check required columns exist
        missing_columns = [col for col in self.config.input_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns for {self.config.name}: {missing_columns}. "
                f"Available: {list(data.columns)}"
            )
        
        # Check sufficient periods
        required_periods = self.get_required_periods()
        if len(data) < required_periods:
            raise ValueError(
                f"Insufficient data for {self.config.name}. "
                f"Required: {required_periods}, Available: {len(data)}"
            )
        
        # Check for NaN values in required columns
        for col in self.config.input_columns:
            if data[col].isna().any():
                self.logger.warning(f"NaN values found in {col} for {self.config.name}")
    
    def _validate_config(self) -> None:
        """
        Validate indicator configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.config.name:
            raise ValueError("Indicator name cannot be empty")
        
        if not self.config.input_columns:
            raise ValueError("Input columns cannot be empty")
        
        if self.config.min_periods < 1:
            raise ValueError("min_periods must be >= 1")
    
    def get_config_dict(self) -> Dict[str, Any]:
        """
        Get indicator configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return {
            'name': self.config.name,
            'type': self.config.indicator_type,
            'parameters': self.config.parameters,
            'input_columns': self.config.input_columns,
            'output_columns': self.config.output_columns,
            'min_periods': self.config.min_periods
        }
    
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply indicator to data with validation.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with indicator columns added
        """
        # Validate input
        self.validate_input(data)
        
        # Calculate indicator
        indicator_data = self.calculate(data)
        
        # Merge with original data
        result = data.copy()
        for col in self.config.output_columns:
            if col in indicator_data.columns:
                result[col] = indicator_data[col]
            else:
                self.logger.warning(f"Expected output column {col} not found in calculation result")
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names this indicator produces.
        
        Returns:
            List of feature column names
        """
        return self.config.output_columns.copy()
    
    def __str__(self) -> str:
        """String representation of indicator."""
        return f"{self.config.indicator_type}({self.config.name})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.config.name}', "
            f"type='{self.config.indicator_type}', "
            f"params={self.config.parameters})"
        )


class CompositeIndicator(BaseIndicator):
    """
    Base class for indicators that combine multiple sub-indicators.
    
    Examples: MACD (combines EMAs), Stochastic (combines multiple calculations)
    """
    
    def __init__(self, config: IndicatorConfig):
        """Initialize composite indicator."""
        super().__init__(config)
        self.sub_indicators: List[BaseIndicator] = []
    
    def add_sub_indicator(self, indicator: BaseIndicator) -> None:
        """
        Add a sub-indicator to this composite.
        
        Args:
            indicator: Sub-indicator to add
        """
        self.sub_indicators.append(indicator)
    
    def get_required_periods(self) -> int:
        """
        Get maximum required periods from all sub-indicators.
        
        Returns:
            Maximum periods needed across all sub-indicators
        """
        if not self.sub_indicators:
            return self.config.min_periods
        
        return max(ind.get_required_periods() for ind in self.sub_indicators)
    
    def validate_input(self, data: pd.DataFrame) -> None:
        """
        Validate input for all sub-indicators.
        
        Args:
            data: Input DataFrame to validate
        """
        super().validate_input(data)
        
        # Validate for all sub-indicators
        for indicator in self.sub_indicators:
            indicator.validate_input(data) 