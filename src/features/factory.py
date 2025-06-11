"""
Feature factory for creating and managing technical indicators.

Provides a centralized factory for creating indicator instances with
proper configuration and validation.
"""

from enum import Enum
from typing import Dict, List, Any, Type, Optional

from .indicators import (
    BaseIndicator,
    # Trend indicators
    MovingAverage, ExponentialMovingAverage, MACD, AverageDirectionalIndex,
    ParabolicSAR, LinearRegression, IchimokuCloud,
    # Momentum indicators
    RSI, StochasticOscillator,
    # Volatility indicators
    BollingerBands, AverageTrueRange, KeltnerChannels, StandardDeviation,
    HistoricalVolatility, DonchianChannels, VIX,
    # Volume indicators
    OnBalanceVolume, VWAP, VolumeWeightedMovingAverage, AccumulationDistributionLine,
    ChaikinMoneyFlow, VolumeOscillator, VolumeRateOfChange, NegativeVolumeIndex,
    PositiveVolumeIndex
)


class IndicatorType(Enum):
    """Enumeration of available indicator types."""
    
    # Trend indicators
    SMA = "simple_moving_average"
    EMA = "exponential_moving_average"
    MACD = "macd"
    ADX = "average_directional_index"
    PSAR = "parabolic_sar"
    LINREG = "linear_regression"
    ICHIMOKU = "ichimoku_cloud"
    
    # Momentum indicators
    RSI = "relative_strength_index"
    STOCH = "stochastic_oscillator"
    
    # Volatility indicators
    BBANDS = "bollinger_bands"
    ATR = "average_true_range"
    KELTNER = "keltner_channels"
    STDDEV = "standard_deviation"
    HVOL = "historical_volatility"
    DONCHIAN = "donchian_channels"
    VIX = "vix_volatility"
    
    # Volume indicators
    OBV = "on_balance_volume"
    VWAP = "volume_weighted_average_price"
    VWMA = "volume_weighted_moving_average"
    AD = "accumulation_distribution"
    CMF = "chaikin_money_flow"
    VOSC = "volume_oscillator"
    VROC = "volume_rate_of_change"
    NVI = "negative_volume_index"
    PVI = "positive_volume_index"


class FeatureFactory:
    """
    Factory for creating technical indicator instances.
    
    Provides a centralized way to create and configure indicators
    with proper validation and parameter management.
    """
    
    def __init__(self):
        """Initialize the feature factory."""
        self._registry: Dict[IndicatorType, Type[BaseIndicator]] = {
            # Trend indicators
            IndicatorType.SMA: MovingAverage,
            IndicatorType.EMA: ExponentialMovingAverage,
            IndicatorType.MACD: MACD,
            IndicatorType.ADX: AverageDirectionalIndex,
            IndicatorType.PSAR: ParabolicSAR,
            IndicatorType.LINREG: LinearRegression,
            IndicatorType.ICHIMOKU: IchimokuCloud,
            
            # Momentum indicators
            IndicatorType.RSI: RSI,
            IndicatorType.STOCH: StochasticOscillator,
            
            # Volatility indicators
            IndicatorType.BBANDS: BollingerBands,
            IndicatorType.ATR: AverageTrueRange,
            IndicatorType.KELTNER: KeltnerChannels,
            IndicatorType.STDDEV: StandardDeviation,
            IndicatorType.HVOL: HistoricalVolatility,
            IndicatorType.DONCHIAN: DonchianChannels,
            IndicatorType.VIX: VIX,
            
            # Volume indicators
            IndicatorType.OBV: OnBalanceVolume,
            IndicatorType.VWAP: VWAP,
            IndicatorType.VWMA: VolumeWeightedMovingAverage,
            IndicatorType.AD: AccumulationDistributionLine,
            IndicatorType.CMF: ChaikinMoneyFlow,
            IndicatorType.VOSC: VolumeOscillator,
            IndicatorType.VROC: VolumeRateOfChange,
            IndicatorType.NVI: NegativeVolumeIndex,
            IndicatorType.PVI: PositiveVolumeIndex
        }
    
    def create_indicator(self, indicator_type: IndicatorType, **kwargs) -> BaseIndicator:
        """
        Create an indicator instance.
        
        Args:
            indicator_type: Type of indicator to create
            **kwargs: Parameters for indicator initialization
            
        Returns:
            Initialized indicator instance
            
        Raises:
            ValueError: If indicator type is not supported
        """
        if indicator_type not in self._registry:
            raise ValueError(f"Unsupported indicator type: {indicator_type}")
        
        indicator_class = self._registry[indicator_type]
        return indicator_class(**kwargs)
    
    def create_indicators(self, configs: List[Dict[str, Any]]) -> List[BaseIndicator]:
        """
        Create multiple indicators from configuration list.
        
        Args:
            configs: List of indicator configurations
                Each config should have 'type' key and optional parameters
                
        Returns:
            List of initialized indicator instances
        """
        indicators = []
        
        for config in configs:
            if 'type' not in config:
                raise ValueError("Indicator config must include 'type' field")
            
            indicator_type = config.pop('type')
            if isinstance(indicator_type, str):
                # Convert string to enum
                indicator_type = IndicatorType(indicator_type)
            
            indicator = self.create_indicator(indicator_type, **config)
            indicators.append(indicator)
        
        return indicators
    
    def get_common_indicators(self, **override_params) -> List[BaseIndicator]:
        """
        Get a standard set of commonly used indicators.
        
        Args:
            **override_params: Parameters to override for specific indicators
                Format: indicator_type_param_name = value
                
        Returns:
            List of common indicator instances
        """
        configs = [
            {'type': IndicatorType.SMA, 'period': 20},
            {'type': IndicatorType.SMA, 'period': 50},
            {'type': IndicatorType.EMA, 'period': 12},
            {'type': IndicatorType.EMA, 'period': 26},
            {'type': IndicatorType.MACD},
            {'type': IndicatorType.RSI, 'period': 14},
            {'type': IndicatorType.BBANDS, 'period': 20, 'std_dev': 2},
            {'type': IndicatorType.ATR, 'period': 14}
        ]
        
        # Apply parameter overrides
        for config in configs:
            indicator_type = config['type']
            for param, value in override_params.items():
                if param.startswith(indicator_type.value):
                    param_name = param.replace(f"{indicator_type.value}_", "")
                    config[param_name] = value
        
        return self.create_indicators(configs)
    
    def get_forex_indicators(self, **override_params) -> List[BaseIndicator]:
        """
        Get indicators optimized for forex trading.
        
        Args:
            **override_params: Parameters to override for specific indicators
            
        Returns:
            List of forex-optimized indicator instances
        """
        configs = [
            # Trend indicators
            {'type': IndicatorType.SMA, 'period': 21},  # 3-week average
            {'type': IndicatorType.EMA, 'period': 13},  # Fibonacci period
            {'type': IndicatorType.EMA, 'period': 34},  # Fibonacci period
            {'type': IndicatorType.MACD, 'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            {'type': IndicatorType.ADX, 'period': 14},
            {'type': IndicatorType.PSAR},
            
            # Momentum indicators
            {'type': IndicatorType.RSI, 'period': 14},
            {'type': IndicatorType.STOCH, 'k_period': 14, 'd_period': 3},
            
            # Volatility indicators
            {'type': IndicatorType.BBANDS, 'period': 20, 'std_dev': 2},
            {'type': IndicatorType.ATR, 'period': 14},
            {'type': IndicatorType.KELTNER, 'period': 20, 'multiplier': 2.0}
        ]
        
        # Apply parameter overrides
        for config in configs:
            indicator_type = config['type']
            for param, value in override_params.items():
                if param.startswith(indicator_type.value):
                    param_name = param.replace(f"{indicator_type.value}_", "")
                    config[param_name] = value
        
        return self.create_indicators(configs)
    
    def get_scalping_indicators(self, **override_params) -> List[BaseIndicator]:
        """
        Get indicators optimized for scalping strategies.
        
        Args:
            **override_params: Parameters to override for specific indicators
            
        Returns:
            List of scalping-optimized indicator instances
        """
        configs = [
            # Fast trend indicators
            {'type': IndicatorType.EMA, 'period': 5},
            {'type': IndicatorType.EMA, 'period': 8},
            {'type': IndicatorType.EMA, 'period': 13},
            
            # Quick momentum
            {'type': IndicatorType.RSI, 'period': 7},
            {'type': IndicatorType.STOCH, 'k_period': 5, 'd_period': 3},
            
            # Volatility for entry/exit
            {'type': IndicatorType.BBANDS, 'period': 10, 'std_dev': 1.5},
            {'type': IndicatorType.ATR, 'period': 7}
        ]
        
        # Apply parameter overrides
        for config in configs:
            indicator_type = config['type']
            for param, value in override_params.items():
                if param.startswith(indicator_type.value):
                    param_name = param.replace(f"{indicator_type.value}_", "")
                    config[param_name] = value
        
        return self.create_indicators(configs)
    
    def validate_compatibility(self, indicators: List[BaseIndicator], 
                             required_columns: List[str]) -> bool:
        """
        Validate that indicators are compatible with available data columns.
        
        Args:
            indicators: List of indicators to validate
            required_columns: Available data columns
            
        Returns:
            True if all indicators are compatible
        """
        for indicator in indicators:
            for required_col in indicator.config.input_columns:
                if required_col not in required_columns:
                    return False
        return True
    
    def get_available_types(self) -> List[IndicatorType]:
        """Get list of all available indicator types."""
        return list(self._registry.keys())
    
    def get_indicator_info(self, indicator_type: IndicatorType) -> Dict[str, Any]:
        """
        Get information about a specific indicator type.
        
        Args:
            indicator_type: Type of indicator to get info for
            
        Returns:
            Dictionary with indicator information
        """
        if indicator_type not in self._registry:
            raise ValueError(f"Unsupported indicator type: {indicator_type}")
        
        indicator_class = self._registry[indicator_type]
        
        # Create a dummy instance to get default config
        try:
            dummy = indicator_class()
            config = dummy.config
        except:
            # If can't create without parameters, provide basic info
            config = None
        
        return {
            'type': indicator_type.value,
            'class': indicator_class.__name__,
            'description': indicator_class.__doc__,
            'config': config.to_dict() if config else None
        }


# Global factory instance
feature_factory = FeatureFactory() 