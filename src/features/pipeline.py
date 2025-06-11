"""
Feature engineering pipeline for comprehensive feature generation.

Orchestrates the entire feature engineering process from data validation
through indicator calculation, feature selection, and output formatting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
from datetime import datetime

from .indicators import BaseIndicator
from .factory import FeatureFactory, IndicatorType
from .calculator import FeatureCalculator, FeatureConfig
try:
    from utils.logger import get_logger
except ImportError:
    # Fallback for when utils module is not available
    def get_logger(name):
        return logging.getLogger(name)


@dataclass
class PipelineConfig:
    """Configuration for feature engineering pipeline."""
    
    # Data configuration
    input_columns: List[str] = field(default_factory=lambda: ['open', 'high', 'low', 'close', 'volume'])
    required_columns: List[str] = field(default_factory=lambda: ['open', 'high', 'low', 'close'])
    datetime_column: str = 'datetime'
    
    # Feature configuration
    enable_price_features: bool = True
    enable_time_features: bool = True
    enable_technical_indicators: bool = True
    enable_volume_features: bool = True
    
    # Indicator configuration
    indicator_configs: Optional[List[Dict[str, Any]]] = None
    use_forex_optimized: bool = True
    use_scalping_indicators: bool = False
    
    # Processing configuration
    min_periods: int = 100
    validation_enabled: bool = True
    feature_selection_enabled: bool = False
    
    # Output configuration
    output_format: str = 'dataframe'  # 'dataframe', 'dict', 'numpy'
    include_metadata: bool = True
    
    # Performance configuration
    chunk_size: Optional[int] = None
    parallel_processing: bool = False
    cache_enabled: bool = True


class FeaturePipeline:
    """
    Comprehensive feature engineering pipeline.
    
    Orchestrates the entire feature engineering process including:
    - Data validation and preprocessing
    - Technical indicator calculation
    - Price and time-based feature generation
    - Feature selection and filtering
    - Output formatting and metadata generation
    """
    
    def __init__(self, config: PipelineConfig = None):
        """
        Initialize feature pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.factory = FeatureFactory()
        
        # Initialize calculator with basic config - it will be reconfigured during feature calculation
        basic_calc_config = FeatureConfig(
            indicators=[],  # Will be populated during setup
            include_price_features=self.config.enable_price_features,
            include_time_features=self.config.enable_time_features
        )
        self.calculator = FeatureCalculator(basic_calc_config)
        
        # Pipeline state
        self.indicators: List[BaseIndicator] = []
        self.feature_names: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self._cache: Dict[str, Any] = {}
        
        # Setup indicators
        self._setup_indicators()
    
    def _setup_indicators(self) -> None:
        """Setup technical indicators based on configuration."""
        if not self.config.enable_technical_indicators:
            return
        
        if self.config.indicator_configs:
            # Use custom indicator configurations
            self.indicators = self.factory.create_indicators(self.config.indicator_configs)
        else:
            # Use pre-configured indicator sets
            if self.config.use_forex_optimized:
                self.indicators = self.factory.get_forex_indicators()
            elif self.config.use_scalping_indicators:
                self.indicators = self.factory.get_scalping_indicators()
            else:
                self.indicators = self.factory.get_common_indicators()
        
        self.logger.info(f"Initialized {len(self.indicators)} technical indicators")
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate input data for feature engineering.
        
        Args:
            data: Input OHLCV data
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required columns
        missing_cols = [col for col in self.config.required_columns if col not in data.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                errors.append(f"Column '{col}' must be numeric")
        
        # Check for sufficient data
        if len(data) < self.config.min_periods:
            errors.append(f"Insufficient data: {len(data)} rows, minimum {self.config.min_periods}")
        
        # Check for null values in critical columns
        for col in self.config.required_columns:
            if col in data.columns and data[col].isnull().any():
                null_count = data[col].isnull().sum()
                errors.append(f"Column '{col}' contains {null_count} null values")
        
        # Check datetime index/column
        if self.config.datetime_column in data.columns:
            try:
                pd.to_datetime(data[self.config.datetime_column])
            except:
                errors.append(f"Invalid datetime format in column '{self.config.datetime_column}'")
        elif not isinstance(data.index, pd.DatetimeIndex):
            errors.append("Data must have datetime index or datetime column")
        
        # Check OHLC logic
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            invalid_ohlc = (
                (data['high'] < data['low']) |
                (data['high'] < data['open']) |
                (data['high'] < data['close']) |
                (data['low'] > data['open']) |
                (data['low'] > data['close'])
            ).any()
            
            if invalid_ohlc:
                errors.append("Invalid OHLC relationships detected")
        
        return len(errors) == 0, errors
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for feature engineering.
        
        Args:
            data: Raw input data
            
        Returns:
            Preprocessed data
        """
        processed_data = data.copy()
        
        # Ensure datetime index
        if self.config.datetime_column in processed_data.columns:
            processed_data[self.config.datetime_column] = pd.to_datetime(
                processed_data[self.config.datetime_column]
            )
            if not isinstance(processed_data.index, pd.DatetimeIndex):
                processed_data = processed_data.set_index(self.config.datetime_column)
        
        # Sort by datetime
        processed_data = processed_data.sort_index()
        
        # Remove duplicates
        processed_data = processed_data[~processed_data.index.duplicated(keep='last')]
        
        # Forward fill small gaps (up to 3 periods)
        for col in self.config.required_columns:
            if col in processed_data.columns:
                # Only forward fill small gaps
                processed_data[col] = processed_data[col].fillna(method='ffill', limit=3)
        
        return processed_data
    
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all features for the given data.
        
        Args:
            data: Preprocessed OHLCV data
            
        Returns:
            DataFrame with all calculated features
        """
        result = data.copy()
        
        try:
            # Use the calculator for price and time features
            if self.config.enable_price_features or self.config.enable_time_features:
                calc_config = FeatureConfig(
                    indicators=[],  # Indicators handled separately
                    include_price_features=self.config.enable_price_features,
                    include_time_features=self.config.enable_time_features
                )
                calculator = FeatureCalculator(calc_config)
                features = calculator.calculate_features(data)
                result = features.copy()
                self.logger.debug(f"Calculated basic features: {len(result.columns)} total columns")
            
            # Calculate technical indicators
            if self.config.enable_technical_indicators and self.indicators:
                for indicator in self.indicators:
                    try:
                        indicator_result = indicator.apply(result)
                        # Only add the new indicator columns, not overwrite existing data
                        for col in indicator.config.output_columns:
                            if col in indicator_result.columns:
                                result[col] = indicator_result[col]
                    except Exception as e:
                        self.logger.warning(f"Failed to calculate {indicator.config.name}: {e}")
                
                self.logger.debug(f"Applied {len(self.indicators)} technical indicators")
            
        except Exception as e:
            self.logger.error(f"Error calculating features: {e}")
            raise
        
        return result
    
    def select_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature selection and filtering.
        
        Args:
            features: Calculated features
            data: Original data for reference
            
        Returns:
            Selected features
        """
        if not self.config.feature_selection_enabled:
            return features
        
        selected_features = features.copy()
        
        # Remove features with too many NaN values (>50%)
        nan_threshold = 0.5
        nan_ratio = selected_features.isnull().mean()
        high_nan_features = nan_ratio[nan_ratio > nan_threshold].index
        
        if len(high_nan_features) > 0:
            selected_features = selected_features.drop(columns=high_nan_features)
            self.logger.info(f"Removed {len(high_nan_features)} features with >50% NaN values")
        
        # Remove constant features
        constant_features = []
        for col in selected_features.columns:
            if selected_features[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            selected_features = selected_features.drop(columns=constant_features)
            self.logger.info(f"Removed {len(constant_features)} constant features")
        
        # Remove highly correlated features (>0.95 correlation)
        correlation_threshold = 0.95
        corr_matrix = selected_features.corr().abs()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        # Remove one feature from each highly correlated pair
        features_to_remove = set()
        for feat1, feat2 in high_corr_pairs:
            if feat1 not in features_to_remove:
                features_to_remove.add(feat2)
        
        if features_to_remove:
            selected_features = selected_features.drop(columns=list(features_to_remove))
            self.logger.info(f"Removed {len(features_to_remove)} highly correlated features")
        
        return selected_features
    
    def generate_metadata(self, features: pd.DataFrame, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate metadata about the feature engineering process.
        
        Args:
            features: Final features
            data: Original data
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            'pipeline_config': self.config.__dict__,
            'generation_timestamp': datetime.now().isoformat(),
            'data_info': {
                'input_shape': data.shape,
                'output_shape': features.shape,
                'date_range': {
                    'start': data.index.min().isoformat() if hasattr(data.index, 'min') else None,
                    'end': data.index.max().isoformat() if hasattr(data.index, 'max') else None
                },
                'input_columns': list(data.columns),
                'output_columns': list(features.columns)
            },
            'indicators': [
                {
                    'name': indicator.config.name,
                    'type': indicator.config.indicator_type,
                    'parameters': indicator.config.parameters,
                    'output_columns': indicator.config.output_columns
                }
                for indicator in self.indicators
            ],
            'feature_stats': {
                'total_features': len(features.columns),
                'null_percentages': features.isnull().mean().to_dict(),
                'feature_types': {
                    'numeric': len([col for col in features.columns if pd.api.types.is_numeric_dtype(features[col])]),
                    'categorical': len([col for col in features.columns if pd.api.types.is_categorical_dtype(features[col])]),
                    'boolean': len([col for col in features.columns if pd.api.types.is_bool_dtype(features[col])])
                }
            }
        }
        
        return metadata
    
    def transform(self, data: pd.DataFrame) -> Union[pd.DataFrame, Dict[str, Any], np.ndarray]:
        """
        Main transformation method to generate features from data.
        
        Args:
            data: Input OHLCV data
            
        Returns:
            Features in specified output format
        """
        self.logger.info(f"Starting feature engineering pipeline for {len(data)} rows")
        
        # Validate data
        if self.config.validation_enabled:
            is_valid, errors = self.validate_data(data)
            if not is_valid:
                raise ValueError(f"Data validation failed: {errors}")
        
        # Preprocess data
        processed_data = self.preprocess_data(data)
        self.logger.debug(f"Preprocessed data: {processed_data.shape}")
        
        # Calculate features
        features = self.calculate_features(processed_data)
        self.logger.info(f"Calculated {len(features.columns)} total features")
        
        # Apply feature selection
        if self.config.feature_selection_enabled:
            features = self.select_features(features, processed_data)
            self.logger.info(f"Selected {len(features.columns)} features after filtering")
        
        # Generate metadata
        if self.config.include_metadata:
            self.metadata = self.generate_metadata(features, data)
        
        # Store feature names
        self.feature_names = list(features.columns)
        
        # Return in specified format
        if self.config.output_format == 'dataframe':
            return features
        elif self.config.output_format == 'dict':
            result = {'features': features}
            if self.config.include_metadata:
                result['metadata'] = self.metadata
            return result
        elif self.config.output_format == 'numpy':
            return features.values
        else:
            raise ValueError(f"Unsupported output format: {self.config.output_format}")
    
    def fit_transform(self, data: pd.DataFrame) -> Union[pd.DataFrame, Dict[str, Any], np.ndarray]:
        """
        Fit the pipeline and transform data (alias for transform).
        
        Args:
            data: Input OHLCV data
            
        Returns:
            Transformed features
        """
        return self.transform(data)
    
    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names."""
        return self.feature_names.copy()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get pipeline metadata."""
        return self.metadata.copy()
    
    def save_config(self, filepath: Union[str, Path]) -> None:
        """
        Save pipeline configuration to file.
        
        Args:
            filepath: Path to save configuration
        """
        import json
        
        config_dict = self.config.__dict__.copy()
        
        # Convert non-serializable objects
        if config_dict.get('indicator_configs'):
            # Convert any enum types to strings
            serializable_configs = []
            for config in config_dict['indicator_configs']:
                serializable_config = config.copy()
                if 'type' in serializable_config and hasattr(serializable_config['type'], 'value'):
                    serializable_config['type'] = serializable_config['type'].value
                serializable_configs.append(serializable_config)
            config_dict['indicator_configs'] = serializable_configs
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"Saved pipeline configuration to {filepath}")
    
    @classmethod
    def load_config(cls, filepath: Union[str, Path]) -> 'FeaturePipeline':
        """
        Load pipeline configuration from file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            FeaturePipeline instance with loaded configuration
        """
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Convert indicator type strings back to enums
        if config_dict.get('indicator_configs'):
            for config in config_dict['indicator_configs']:
                if 'type' in config and isinstance(config['type'], str):
                    config['type'] = IndicatorType(config['type'])
        
        config = PipelineConfig(**config_dict)
        return cls(config)


# Convenience function for quick feature generation
def generate_features(data: pd.DataFrame, 
                     config: PipelineConfig = None,
                     **kwargs) -> pd.DataFrame:
    """
    Convenience function to generate features with default pipeline.
    
    Args:
        data: Input OHLCV data
        config: Optional pipeline configuration
        **kwargs: Additional configuration parameters
        
    Returns:
        Generated features DataFrame
    """
    if config is None:
        config = PipelineConfig(**kwargs)
    
    pipeline = FeaturePipeline(config)
    return pipeline.transform(data) 