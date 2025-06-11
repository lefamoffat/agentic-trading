"""
Feature engineering framework for trading system.

This package contains:
- Technical indicators library
- Feature calculators and processors
- Multi-timeframe feature generation
- Market analysis tools
"""

# Import main components
from .factory import FeatureFactory, IndicatorType, feature_factory
from .calculator import FeatureCalculator, FeatureConfig
from .pipeline import FeaturePipeline, PipelineConfig, generate_features

# Import indicator base classes
from .indicators import BaseIndicator, CompositeIndicator, IndicatorConfig

# Import specific indicator categories (for advanced usage)
from .indicators.trend import (
    MovingAverage, ExponentialMovingAverage, MACD, AverageDirectionalIndex,
    ParabolicSAR, LinearRegression, IchimokuCloud
)
from .indicators.momentum import RSI, StochasticOscillator
from .indicators.volatility import (
    BollingerBands, AverageTrueRange, KeltnerChannels, StandardDeviation,
    HistoricalVolatility, DonchianChannels, VIX,
)
from .indicators.volume import (
    OnBalanceVolume, VWAP, VolumeWeightedMovingAverage, AccumulationDistributionLine,
    ChaikinMoneyFlow, VolumeOscillator, VolumeRateOfChange, NegativeVolumeIndex,
    PositiveVolumeIndex
)

__all__ = [
    # Main pipeline components
    'FeaturePipeline',
    'PipelineConfig', 
    'generate_features',
    
    # Factory and calculator
    'FeatureFactory',
    'IndicatorType',
    'feature_factory',
    'FeatureCalculator',
    'FeatureConfig',
    
    # Base classes
    'BaseIndicator',
    'CompositeIndicator',
    'IndicatorConfig',
    
    # Trend indicators
    'MovingAverage',
    'ExponentialMovingAverage',
    'MACD',
    'AverageDirectionalIndex',
    'ParabolicSAR',
    'LinearRegression',
    'IchimokuCloud',
    
    # Momentum indicators
    'RSI',
    'StochasticOscillator',
    
    # Volatility indicators
    'BollingerBands',
    'AverageTrueRange',
    'KeltnerChannels',
    'StandardDeviation',
    'HistoricalVolatility',
    'DonchianChannels',
    'VIX',
    
    # Volume indicators
    'OnBalanceVolume',
    'VWAP',
    'VolumeWeightedMovingAverage',
    'AccumulationDistributionLine',
    'ChaikinMoneyFlow',
    'VolumeOscillator',
    'VolumeRateOfChange',
    'NegativeVolumeIndex',
    'PositiveVolumeIndex'
] 