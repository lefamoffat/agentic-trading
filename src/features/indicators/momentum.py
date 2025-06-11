"""
Momentum-based technical indicators.

Includes RSI, Stochastic Oscillator, and other momentum oscillators
commonly used in forex trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from .base import BaseIndicator, IndicatorConfig


class RSI(BaseIndicator):
    """
    Relative Strength Index (RSI) indicator.
    
    Measures the speed and change of price movements.
    """
    
    def __init__(self, period: int = 14, column: str = 'close', name: str = None):
        """
        Initialize RSI indicator.
        
        Args:
            period: Number of periods for calculation
            column: Column to calculate on (default: 'close')
            name: Custom name for the indicator
        """
        if name is None:
            name = f"rsi_{period}"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="RSI",
            parameters={'period': period},
            input_columns=[column],
            output_columns=[name],
            min_periods=period + 1
        )
        super().__init__(config)
        self.period = period
        self.column = column
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with RSI values
        """
        result = pd.DataFrame(index=data.index)
        
        # Calculate price changes
        prices = data[self.column]
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses using Wilder's smoothing
        avg_gains = gains.ewm(alpha=1/self.period, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1/self.period, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        result[self.config.name] = rsi
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return self.period + 1


class StochasticOscillator(BaseIndicator):
    """
    Stochastic Oscillator indicator.
    
    Compares closing price to the high-low range over a given period.
    """
    
    def __init__(self, k_period: int = 14, d_period: int = 3, 
                 smooth_k: int = 3, name: str = None):
        """
        Initialize Stochastic Oscillator.
        
        Args:
            k_period: Period for %K calculation
            d_period: Period for %D (signal line) calculation  
            smooth_k: Smoothing period for %K
            name: Custom name prefix for the indicator
        """
        if name is None:
            name = "stoch"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="STOCH",
            parameters={
                'k_period': k_period,
                'd_period': d_period,
                'smooth_k': smooth_k
            },
            input_columns=['high', 'low', 'close'],
            output_columns=[f"{name}_k", f"{name}_d"],
            min_periods=k_period + smooth_k + d_period
        )
        super().__init__(config)
        self.k_period = k_period
        self.d_period = d_period
        self.smooth_k = smooth_k
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with %K and %D values
        """
        result = pd.DataFrame(index=data.index)
        
        # Calculate raw %K
        lowest_low = data['low'].rolling(window=self.k_period).min()
        highest_high = data['high'].rolling(window=self.k_period).max()
        
        raw_k = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
        
        # Smooth %K
        k_percent = raw_k.rolling(window=self.smooth_k).mean()
        
        # Calculate %D (signal line)
        d_percent = k_percent.rolling(window=self.d_period).mean()
        
        result[f"{self.config.name}_k"] = k_percent
        result[f"{self.config.name}_d"] = d_percent
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return self.k_period + self.smooth_k + self.d_period


class WilliamsR(BaseIndicator):
    """
    Williams %R indicator.
    
    Measures overbought and oversold levels.
    """
    
    def __init__(self, period: int = 14, name: str = None):
        """
        Initialize Williams %R.
        
        Args:
            period: Number of periods for calculation
            name: Custom name for the indicator
        """
        if name is None:
            name = f"williams_r_{period}"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="WILLIAMS_R",
            parameters={'period': period},
            input_columns=['high', 'low', 'close'],
            output_columns=[name],
            min_periods=period
        )
        super().__init__(config)
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Williams %R.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with Williams %R values
        """
        result = pd.DataFrame(index=data.index)
        
        # Calculate highest high and lowest low over period
        highest_high = data['high'].rolling(window=self.period).max()
        lowest_low = data['low'].rolling(window=self.period).min()
        
        # Calculate Williams %R
        williams_r = -100 * (highest_high - data['close']) / (highest_high - lowest_low)
        
        result[self.config.name] = williams_r
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return self.period


class CommodityChannelIndex(BaseIndicator):
    """
    Commodity Channel Index (CCI) indicator.
    
    Measures deviation from the average price.
    """
    
    def __init__(self, period: int = 20, name: str = None):
        """
        Initialize CCI.
        
        Args:
            period: Number of periods for calculation
            name: Custom name for the indicator
        """
        if name is None:
            name = f"cci_{period}"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="CCI",
            parameters={'period': period},
            input_columns=['high', 'low', 'close'],
            output_columns=[name],
            min_periods=period
        )
        super().__init__(config)
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate CCI.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with CCI values
        """
        result = pd.DataFrame(index=data.index)
        
        # Calculate Typical Price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate Simple Moving Average of Typical Price
        sma_tp = typical_price.rolling(window=self.period).mean()
        
        # Calculate Mean Deviation
        mad = typical_price.rolling(window=self.period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        
        # Calculate CCI
        cci = (typical_price - sma_tp) / (0.015 * mad)
        
        result[self.config.name] = cci
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return self.period


class RateOfChange(BaseIndicator):
    """
    Rate of Change (ROC) indicator.
    
    Measures the percentage change in price over a specified period.
    """
    
    def __init__(self, period: int = 10, column: str = 'close', name: str = None):
        """
        Initialize ROC.
        
        Args:
            period: Number of periods for calculation
            column: Column to calculate on (default: 'close')
            name: Custom name for the indicator
        """
        if name is None:
            name = f"roc_{period}"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="ROC",
            parameters={'period': period},
            input_columns=[column],
            output_columns=[name],
            min_periods=period + 1
        )
        super().__init__(config)
        self.period = period
        self.column = column
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ROC.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with ROC values
        """
        result = pd.DataFrame(index=data.index)
        
        # Calculate Rate of Change
        current_price = data[self.column]
        previous_price = data[self.column].shift(self.period)
        
        roc = 100 * (current_price - previous_price) / previous_price
        
        result[self.config.name] = roc
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return self.period + 1


class MoneyFlowIndex(BaseIndicator):
    """
    Money Flow Index (MFI) indicator.
    
    Volume-weighted version of RSI.
    """
    
    def __init__(self, period: int = 14, name: str = None):
        """
        Initialize MFI.
        
        Args:
            period: Number of periods for calculation
            name: Custom name for the indicator
        """
        if name is None:
            name = f"mfi_{period}"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="MFI",
            parameters={'period': period},
            input_columns=['high', 'low', 'close', 'volume'],
            output_columns=[name],
            min_periods=period + 1
        )
        super().__init__(config)
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MFI.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with MFI values
        """
        result = pd.DataFrame(index=data.index)
        
        # Calculate Typical Price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate Raw Money Flow
        money_flow = typical_price * data['volume']
        
        # Identify positive and negative money flow
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        # Calculate Money Flow Ratio
        positive_flow_sum = positive_flow.rolling(window=self.period).sum()
        negative_flow_sum = negative_flow.rolling(window=self.period).sum()
        
        money_ratio = positive_flow_sum / negative_flow_sum
        
        # Calculate MFI
        mfi = 100 - (100 / (1 + money_ratio))
        
        result[self.config.name] = mfi
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return self.period + 1 