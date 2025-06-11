"""
Volume-based technical indicators.

Includes OBV, VWAP, and other volume-based indicators
commonly used in forex trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from .base import BaseIndicator, IndicatorConfig


class OnBalanceVolume(BaseIndicator):
    """
    On-Balance Volume (OBV) indicator.
    
    Relates volume to price change.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize OBV.
        
        Args:
            name: Custom name for the indicator
        """
        if name is None:
            name = "obv"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="OBV",
            parameters={},
            input_columns=['close', 'volume'],
            output_columns=[name],
            min_periods=2
        )
        super().__init__(config)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate OBV.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with OBV values
        """
        result = pd.DataFrame(index=data.index)
        
        # Calculate price change direction
        price_change = data['close'].diff()
        
        # Assign volume based on price direction
        obv_volume = data['volume'].copy()
        obv_volume[price_change < 0] = -obv_volume[price_change < 0]
        obv_volume[price_change == 0] = 0
        
        # Calculate cumulative OBV
        obv = obv_volume.cumsum()
        
        result[self.config.name] = obv
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return 2


class VWAP(BaseIndicator):
    """
    Volume Weighted Average Price (VWAP) indicator.
    
    Calculates the average price weighted by volume.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize VWAP.
        
        Args:
            name: Custom name for the indicator
        """
        if name is None:
            name = "vwap"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="VWAP",
            parameters={},
            input_columns=['high', 'low', 'close', 'volume'],
            output_columns=[name],
            min_periods=1
        )
        super().__init__(config)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWAP.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with VWAP values
        """
        result = pd.DataFrame(index=data.index)
        
        # Calculate typical price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate cumulative volume and cumulative volume * typical price
        cumulative_volume = data['volume'].cumsum()
        cumulative_price_volume = (typical_price * data['volume']).cumsum()
        
        # Calculate VWAP
        vwap = cumulative_price_volume / cumulative_volume
        
        result[self.config.name] = vwap
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return 1


class VolumeWeightedMovingAverage(BaseIndicator):
    """
    Volume Weighted Moving Average (VWMA) indicator.
    
    Moving average that gives more weight to periods with higher volume.
    """
    
    def __init__(self, period: int = 20, name: str = None):
        """
        Initialize VWMA.
        
        Args:
            period: Number of periods for calculation
            name: Custom name for the indicator
        """
        if name is None:
            name = f"vwma_{period}"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="VWMA",
            parameters={'period': period},
            input_columns=['close', 'volume'],
            output_columns=[name],
            min_periods=period
        )
        super().__init__(config)
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWMA.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with VWMA values
        """
        result = pd.DataFrame(index=data.index)
        
        # Calculate rolling sums
        price_volume = (data['close'] * data['volume']).rolling(window=self.period).sum()
        volume_sum = data['volume'].rolling(window=self.period).sum()
        
        # Calculate VWMA
        vwma = price_volume / volume_sum
        
        result[self.config.name] = vwma
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return self.period


class AccumulationDistributionLine(BaseIndicator):
    """
    Accumulation/Distribution Line (A/D Line) indicator.
    
    Measures the cumulative flow of money into and out of a security.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize A/D Line.
        
        Args:
            name: Custom name for the indicator
        """
        if name is None:
            name = "ad_line"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="AD_LINE",
            parameters={},
            input_columns=['high', 'low', 'close', 'volume'],
            output_columns=[name],
            min_periods=1
        )
        super().__init__(config)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate A/D Line.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with A/D Line values
        """
        result = pd.DataFrame(index=data.index)
        
        # Calculate Money Flow Multiplier
        clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        
        # Handle division by zero (when high = low)
        clv = clv.fillna(0)
        
        # Calculate Money Flow Volume
        money_flow_volume = clv * data['volume']
        
        # Calculate cumulative A/D Line
        ad_line = money_flow_volume.cumsum()
        
        result[self.config.name] = ad_line
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return 1


class ChaikinMoneyFlow(BaseIndicator):
    """
    Chaikin Money Flow (CMF) indicator.
    
    Measures the amount of Money Flow Volume over a specific period.
    """
    
    def __init__(self, period: int = 20, name: str = None):
        """
        Initialize CMF.
        
        Args:
            period: Number of periods for calculation
            name: Custom name for the indicator
        """
        if name is None:
            name = f"cmf_{period}"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="CMF",
            parameters={'period': period},
            input_columns=['high', 'low', 'close', 'volume'],
            output_columns=[name],
            min_periods=period
        )
        super().__init__(config)
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate CMF.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with CMF values
        """
        result = pd.DataFrame(index=data.index)
        
        # Calculate Money Flow Multiplier
        clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        
        # Handle division by zero
        clv = clv.fillna(0)
        
        # Calculate Money Flow Volume
        money_flow_volume = clv * data['volume']
        
        # Calculate CMF over the period
        cmf = (money_flow_volume.rolling(window=self.period).sum() / 
               data['volume'].rolling(window=self.period).sum())
        
        result[self.config.name] = cmf
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return self.period


class VolumeOscillator(BaseIndicator):
    """
    Volume Oscillator indicator.
    
    Shows the relationship between two volume moving averages.
    """
    
    def __init__(self, short_period: int = 5, long_period: int = 10, name: str = None):
        """
        Initialize Volume Oscillator.
        
        Args:
            short_period: Short period for moving average
            long_period: Long period for moving average
            name: Custom name for the indicator
        """
        if name is None:
            name = f"vol_osc_{short_period}_{long_period}"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="VOL_OSC",
            parameters={'short_period': short_period, 'long_period': long_period},
            input_columns=['volume'],
            output_columns=[name],
            min_periods=long_period
        )
        super().__init__(config)
        self.short_period = short_period
        self.long_period = long_period
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Volume Oscillator.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with Volume Oscillator values
        """
        result = pd.DataFrame(index=data.index)
        
        # Calculate short and long moving averages of volume
        short_ma = data['volume'].rolling(window=self.short_period).mean()
        long_ma = data['volume'].rolling(window=self.long_period).mean()
        
        # Calculate oscillator as percentage difference
        vol_osc = ((short_ma - long_ma) / long_ma) * 100
        
        result[self.config.name] = vol_osc
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return self.long_period


class VolumeRateOfChange(BaseIndicator):
    """
    Volume Rate of Change (VROC) indicator.
    
    Measures the rate of change in volume.
    """
    
    def __init__(self, period: int = 10, name: str = None):
        """
        Initialize VROC.
        
        Args:
            period: Number of periods for calculation
            name: Custom name for the indicator
        """
        if name is None:
            name = f"vroc_{period}"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="VROC",
            parameters={'period': period},
            input_columns=['volume'],
            output_columns=[name],
            min_periods=period + 1
        )
        super().__init__(config)
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VROC.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with VROC values
        """
        result = pd.DataFrame(index=data.index)
        
        # Calculate volume rate of change
        current_volume = data['volume']
        previous_volume = data['volume'].shift(self.period)
        
        vroc = ((current_volume - previous_volume) / previous_volume) * 100
        
        result[self.config.name] = vroc
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return self.period + 1


class NegativeVolumeIndex(BaseIndicator):
    """
    Negative Volume Index (NVI) indicator.
    
    Focuses on periods when volume decreases.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize NVI.
        
        Args:
            name: Custom name for the indicator
        """
        if name is None:
            name = "nvi"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="NVI",
            parameters={},
            input_columns=['close', 'volume'],
            output_columns=[name],
            min_periods=2
        )
        super().__init__(config)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate NVI.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with NVI values
        """
        result = pd.DataFrame(index=data.index)
        
        # Initialize NVI
        nvi = pd.Series(index=data.index, dtype=float)
        nvi.iloc[0] = 1000  # Starting value
        
        # Calculate price change percentage
        price_change = data['close'].pct_change()
        volume_change = data['volume'].diff()
        
        # Update NVI only on days when volume decreases
        for i in range(1, len(data)):
            if volume_change.iloc[i] < 0:
                nvi.iloc[i] = nvi.iloc[i-1] * (1 + price_change.iloc[i])
            else:
                nvi.iloc[i] = nvi.iloc[i-1]
        
        result[self.config.name] = nvi
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return 2


class PositiveVolumeIndex(BaseIndicator):
    """
    Positive Volume Index (PVI) indicator.
    
    Focuses on periods when volume increases.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize PVI.
        
        Args:
            name: Custom name for the indicator
        """
        if name is None:
            name = "pvi"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="PVI",
            parameters={},
            input_columns=['close', 'volume'],
            output_columns=[name],
            min_periods=2
        )
        super().__init__(config)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate PVI.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with PVI values
        """
        result = pd.DataFrame(index=data.index)
        
        # Initialize PVI
        pvi = pd.Series(index=data.index, dtype=float)
        pvi.iloc[0] = 1000  # Starting value
        
        # Calculate price change percentage
        price_change = data['close'].pct_change()
        volume_change = data['volume'].diff()
        
        # Update PVI only on days when volume increases
        for i in range(1, len(data)):
            if volume_change.iloc[i] > 0:
                pvi.iloc[i] = pvi.iloc[i-1] * (1 + price_change.iloc[i])
            else:
                pvi.iloc[i] = pvi.iloc[i-1]
        
        result[self.config.name] = pvi
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return 2 