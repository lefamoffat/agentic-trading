"""
Volatility-based technical indicators.

Includes Bollinger Bands, ATR, and other volatility measures
commonly used in forex trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from .base import BaseIndicator, IndicatorConfig


class BollingerBands(BaseIndicator):
    """
    Bollinger Bands indicator.
    
    Provides upper and lower bands based on standard deviation around a moving average.
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, 
                 column: str = 'close', name: str = None):
        """
        Initialize Bollinger Bands.
        
        Args:
            period: Number of periods for calculation
            std_dev: Number of standard deviations for bands
            column: Column to calculate on (default: 'close')
            name: Custom name prefix for the indicator
        """
        if name is None:
            name = "bb"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="BOLLINGER_BANDS",
            parameters={'period': period, 'std_dev': std_dev},
            input_columns=[column],
            output_columns=[f"{name}_upper", f"{name}_middle", f"{name}_lower", f"{name}_width", f"{name}_percent"],
            min_periods=period
        )
        super().__init__(config)
        self.period = period
        self.std_dev = std_dev
        self.column = column
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with Bollinger Bands values
        """
        result = pd.DataFrame(index=data.index)
        
        # Calculate middle band (SMA)
        middle_band = data[self.column].rolling(window=self.period).mean()
        
        # Calculate standard deviation
        std = data[self.column].rolling(window=self.period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (self.std_dev * std)
        lower_band = middle_band - (self.std_dev * std)
        
        # Calculate bandwidth and %B
        bandwidth = (upper_band - lower_band) / middle_band
        percent_b = (data[self.column] - lower_band) / (upper_band - lower_band)
        
        result[f"{self.config.name}_upper"] = upper_band
        result[f"{self.config.name}_middle"] = middle_band
        result[f"{self.config.name}_lower"] = lower_band
        result[f"{self.config.name}_width"] = bandwidth
        result[f"{self.config.name}_percent"] = percent_b
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return self.period


class AverageTrueRange(BaseIndicator):
    """
    Average True Range (ATR) indicator.
    
    Measures market volatility by calculating the average of true ranges.
    """
    
    def __init__(self, period: int = 14, name: str = None):
        """
        Initialize ATR.
        
        Args:
            period: Number of periods for calculation
            name: Custom name for the indicator
        """
        if name is None:
            name = f"atr_{period}"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="ATR",
            parameters={'period': period},
            input_columns=['high', 'low', 'close'],
            output_columns=[name],
            min_periods=period
        )
        super().__init__(config)
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ATR.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with ATR values
        """
        result = pd.DataFrame(index=data.index)
        
        # Calculate True Range components
        high = data['high']
        low = data['low']
        close = data['close']
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        # True Range is the maximum of the three components
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR using Wilder's smoothing
        atr = true_range.ewm(alpha=1/self.period, adjust=False).mean()
        
        result[self.config.name] = atr
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return self.period


class KeltnerChannels(BaseIndicator):
    """
    Keltner Channels indicator.
    
    Similar to Bollinger Bands but uses ATR instead of standard deviation.
    """
    
    def __init__(self, period: int = 20, atr_period: int = 10, 
                 multiplier: float = 2.0, name: str = None):
        """
        Initialize Keltner Channels.
        
        Args:
            period: Period for EMA calculation
            atr_period: Period for ATR calculation
            multiplier: ATR multiplier for bands
            name: Custom name prefix for the indicator
        """
        if name is None:
            name = "kc"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="KELTNER_CHANNELS",
            parameters={
                'period': period,
                'atr_period': atr_period,
                'multiplier': multiplier
            },
            input_columns=['high', 'low', 'close'],
            output_columns=[f"{name}_upper", f"{name}_middle", f"{name}_lower"],
            min_periods=max(period, atr_period)
        )
        super().__init__(config)
        self.period = period
        self.atr_period = atr_period
        self.multiplier = multiplier
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Keltner Channels.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with Keltner Channels values
        """
        result = pd.DataFrame(index=data.index)
        
        # Calculate middle line (EMA of close)
        middle_line = data['close'].ewm(span=self.period).mean()
        
        # Calculate ATR
        atr_indicator = AverageTrueRange(period=self.atr_period, name="temp_atr")
        atr_data = atr_indicator.calculate(data)
        atr = atr_data["temp_atr"]
        
        # Calculate upper and lower bands
        upper_band = middle_line + (self.multiplier * atr)
        lower_band = middle_line - (self.multiplier * atr)
        
        result[f"{self.config.name}_upper"] = upper_band
        result[f"{self.config.name}_middle"] = middle_line
        result[f"{self.config.name}_lower"] = lower_band
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return max(self.period, self.atr_period)


class StandardDeviation(BaseIndicator):
    """
    Standard Deviation indicator.
    
    Measures the volatility of price movements.
    """
    
    def __init__(self, period: int = 20, column: str = 'close', name: str = None):
        """
        Initialize Standard Deviation.
        
        Args:
            period: Number of periods for calculation
            column: Column to calculate on (default: 'close')
            name: Custom name for the indicator
        """
        if name is None:
            name = f"std_{period}"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="STD_DEV",
            parameters={'period': period},
            input_columns=[column],
            output_columns=[name],
            min_periods=period
        )
        super().__init__(config)
        self.period = period
        self.column = column
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Standard Deviation.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with Standard Deviation values
        """
        result = pd.DataFrame(index=data.index)
        
        std_dev = data[self.column].rolling(window=self.period).std()
        
        result[self.config.name] = std_dev
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return self.period


class HistoricalVolatility(BaseIndicator):
    """
    Historical Volatility indicator.
    
    Measures annualized volatility based on log returns.
    """
    
    def __init__(self, period: int = 20, column: str = 'close', 
                 trading_periods: int = 252, name: str = None):
        """
        Initialize Historical Volatility.
        
        Args:
            period: Number of periods for calculation
            column: Column to calculate on (default: 'close')
            trading_periods: Number of trading periods per year (252 for daily, 8760 for hourly)
            name: Custom name for the indicator
        """
        if name is None:
            name = f"hvol_{period}"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="HIST_VOL",
            parameters={'period': period, 'trading_periods': trading_periods},
            input_columns=[column],
            output_columns=[name],
            min_periods=period + 1
        )
        super().__init__(config)
        self.period = period
        self.column = column
        self.trading_periods = trading_periods
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Historical Volatility.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with Historical Volatility values
        """
        result = pd.DataFrame(index=data.index)
        
        # Calculate log returns
        log_returns = np.log(data[self.column] / data[self.column].shift(1))
        
        # Calculate rolling standard deviation of log returns
        vol = log_returns.rolling(window=self.period).std()
        
        # Annualize volatility
        annualized_vol = vol * np.sqrt(self.trading_periods) * 100
        
        result[self.config.name] = annualized_vol
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return self.period + 1


class DonchianChannels(BaseIndicator):
    """
    Donchian Channels indicator.
    
    Uses the highest high and lowest low over a period to create channels.
    """
    
    def __init__(self, period: int = 20, name: str = None):
        """
        Initialize Donchian Channels.
        
        Args:
            period: Number of periods for calculation
            name: Custom name prefix for the indicator
        """
        if name is None:
            name = "dc"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="DONCHIAN_CHANNELS",
            parameters={'period': period},
            input_columns=['high', 'low', 'close'],
            output_columns=[f"{name}_upper", f"{name}_middle", f"{name}_lower"],
            min_periods=period
        )
        super().__init__(config)
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Donchian Channels.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with Donchian Channels values
        """
        result = pd.DataFrame(index=data.index)
        
        # Calculate upper and lower bands
        upper_band = data['high'].rolling(window=self.period).max()
        lower_band = data['low'].rolling(window=self.period).min()
        
        # Calculate middle band (average of upper and lower)
        middle_band = (upper_band + lower_band) / 2
        
        result[f"{self.config.name}_upper"] = upper_band
        result[f"{self.config.name}_middle"] = middle_band
        result[f"{self.config.name}_lower"] = lower_band
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return self.period


class VIX(BaseIndicator):
    """
    VIX-style volatility indicator.
    
    Calculates implied volatility from option-like behavior of price movements.
    Note: This is a simplified version for spot markets.
    """
    
    def __init__(self, period: int = 30, column: str = 'close', name: str = None):
        """
        Initialize VIX-style indicator.
        
        Args:
            period: Number of periods for calculation
            column: Column to calculate on (default: 'close')
            name: Custom name for the indicator
        """
        if name is None:
            name = f"vix_{period}"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="VIX",
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
        Calculate VIX-style volatility.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with VIX values
        """
        result = pd.DataFrame(index=data.index)
        
        # Calculate returns
        returns = data[self.column].pct_change()
        
        # Calculate realized volatility (rolling standard deviation)
        realized_vol = returns.rolling(window=self.period).std()
        
        # Calculate mean reversion component
        price_sma = data[self.column].rolling(window=self.period).mean()
        distance_from_mean = ((data[self.column] - price_sma) / price_sma).abs()
        
        # Combine realized volatility with mean reversion
        vix_value = realized_vol * (1 + distance_from_mean) * 100
        
        result[self.config.name] = vix_value
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return self.period + 1 