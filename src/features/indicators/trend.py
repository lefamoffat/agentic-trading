"""
Trend-based technical indicators.

Includes moving averages, MACD, ADX, and other trend-following indicators
commonly used in forex trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from .base import BaseIndicator, CompositeIndicator, IndicatorConfig


class MovingAverage(BaseIndicator):
    """Simple Moving Average (SMA) indicator."""
    
    def __init__(self, period: int = 20, column: str = 'close', name: str = None):
        """
        Initialize Simple Moving Average.
        
        Args:
            period: Number of periods for calculation
            column: Column to calculate on (default: 'close')
            name: Custom name for the indicator
        """
        if name is None:
            name = f"sma_{period}"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="SMA",
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
        Calculate Simple Moving Average.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with SMA values
        """
        result = pd.DataFrame(index=data.index)
        
        sma = data[self.column].rolling(window=self.period).mean()
        
        result[self.config.name] = sma
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return self.period


class ExponentialMovingAverage(BaseIndicator):
    """Exponential Moving Average (EMA) indicator."""
    
    def __init__(self, period: int = 20, column: str = 'close', name: str = None):
        """
        Initialize Exponential Moving Average.
        
        Args:
            period: Number of periods for calculation
            column: Column to calculate on (default: 'close')
            name: Custom name for the indicator
        """
        if name is None:
            name = f"ema_{period}"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="EMA",
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
        Calculate Exponential Moving Average.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with EMA values
        """
        result = pd.DataFrame(index=data.index)
        
        ema = data[self.column].ewm(span=self.period, adjust=False).mean()
        
        result[self.config.name] = ema
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return self.period


class MACD(CompositeIndicator):
    """
    MACD (Moving Average Convergence Divergence) indicator.
    
    Consists of MACD line, signal line, and histogram.
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, 
                 signal_period: int = 9, column: str = 'close', name: str = None):
        """
        Initialize MACD indicator.
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            column: Column to calculate on (default: 'close')
            name: Custom name prefix for the indicator
        """
        if name is None:
            name = "macd"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="MACD",
            parameters={
                'fast_period': fast_period,
                'slow_period': slow_period,
                'signal_period': signal_period
            },
            input_columns=[column],
            output_columns=[f"{name}_line", f"{name}_signal", f"{name}_histogram"],
            min_periods=slow_period + signal_period
        )
        super().__init__(config)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.column = column
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with MACD values
        """
        result = pd.DataFrame(index=data.index)
        
        # Calculate fast and slow EMAs
        fast_ema = data[self.column].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = data[self.column].ewm(span=self.slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        result[f"{self.config.name}_line"] = macd_line
        result[f"{self.config.name}_signal"] = signal_line
        result[f"{self.config.name}_histogram"] = histogram
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return self.slow_period + self.signal_period


class AverageDirectionalIndex(BaseIndicator):
    """
    Average Directional Index (ADX) indicator.
    
    Measures the strength of a trend without regard to direction.
    """
    
    def __init__(self, period: int = 14, name: str = None):
        """
        Initialize ADX indicator.
        
        Args:
            period: Number of periods for calculation
            name: Custom name for the indicator
        """
        if name is None:
            name = f"adx_{period}"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="ADX",
            parameters={'period': period},
            input_columns=['high', 'low', 'close'],
            output_columns=[name, f"di_plus_{period}", f"di_minus_{period}"],
            min_periods=period * 2
        )
        super().__init__(config)
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ADX.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with ADX values
        """
        result = pd.DataFrame(index=data.index)
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate directional movements
        plus_dm = high.diff()
        minus_dm = low.diff() * -1
        
        # Only keep positive movements
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Zero out smaller movement when both are positive
        plus_dm[(plus_dm < minus_dm)] = 0
        minus_dm[(minus_dm < plus_dm)] = 0
        
        # Calculate smoothed True Range and Directional Movements
        tr_smooth = tr.ewm(alpha=1/self.period, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=1/self.period, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1/self.period, adjust=False).mean()
        
        # Calculate Directional Indicators
        di_plus = (plus_dm_smooth / tr_smooth) * 100
        di_minus = (minus_dm_smooth / tr_smooth) * 100
        
        # Calculate DX
        dx = ((di_plus - di_minus).abs() / (di_plus + di_minus)) * 100
        
        # Calculate ADX
        adx = dx.ewm(alpha=1/self.period, adjust=False).mean()
        
        result[self.config.name] = adx
        result[f"di_plus_{self.period}"] = di_plus
        result[f"di_minus_{self.period}"] = di_minus
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return self.period * 2


class ParabolicSAR(BaseIndicator):
    """
    Parabolic SAR (Stop and Reverse) indicator.
    
    Provides potential reversal points in price direction.
    """
    
    def __init__(self, acceleration: float = 0.02, maximum: float = 0.20, name: str = None):
        """
        Initialize Parabolic SAR.
        
        Args:
            acceleration: Acceleration factor step
            maximum: Maximum acceleration factor
            name: Custom name for the indicator
        """
        if name is None:
            name = "psar"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="PSAR",
            parameters={'acceleration': acceleration, 'maximum': maximum},
            input_columns=['high', 'low'],
            output_columns=[name, f"{name}_trend"],
            min_periods=2
        )
        super().__init__(config)
        self.acceleration = acceleration
        self.maximum = maximum
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Parabolic SAR.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with PSAR values
        """
        result = pd.DataFrame(index=data.index)
        
        high = data['high'].values
        low = data['low'].values
        
        # Initialize arrays
        psar = np.zeros(len(high))
        trend = np.zeros(len(high))  # 1 for uptrend, -1 for downtrend
        af = np.zeros(len(high))
        ep = np.zeros(len(high))  # Extreme point
        
        # Initialize first values
        psar[0] = low[0]
        trend[0] = 1
        af[0] = self.acceleration
        ep[0] = high[0]
        
        for i in range(1, len(high)):
            if trend[i-1] == 1:  # Uptrend
                psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
                
                # Check if trend should reverse
                if low[i] <= psar[i]:
                    trend[i] = -1
                    psar[i] = ep[i-1]
                    ep[i] = low[i]
                    af[i] = self.acceleration
                else:
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + self.acceleration, self.maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                    
                    # Ensure PSAR doesn't exceed recent lows
                    psar[i] = min(psar[i], low[i-1])
                    if i > 1:
                        psar[i] = min(psar[i], low[i-2])
            
            else:  # Downtrend
                psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
                
                # Check if trend should reverse
                if high[i] >= psar[i]:
                    trend[i] = 1
                    psar[i] = ep[i-1]
                    ep[i] = high[i]
                    af[i] = self.acceleration
                else:
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + self.acceleration, self.maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                    
                    # Ensure PSAR doesn't exceed recent highs
                    psar[i] = max(psar[i], high[i-1])
                    if i > 1:
                        psar[i] = max(psar[i], high[i-2])
        
        result[self.config.name] = psar
        result[f"{self.config.name}_trend"] = trend
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return 2


class LinearRegression(BaseIndicator):
    """
    Linear Regression indicator.
    
    Calculates linear regression line over a specified period.
    """
    
    def __init__(self, period: int = 20, column: str = 'close', name: str = None):
        """
        Initialize Linear Regression.
        
        Args:
            period: Number of periods for calculation
            column: Column to calculate on (default: 'close')
            name: Custom name for the indicator
        """
        if name is None:
            name = f"linreg_{period}"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="LINREG",
            parameters={'period': period},
            input_columns=[column],
            output_columns=[name, f"{name}_slope", f"{name}_r2"],
            min_periods=period
        )
        super().__init__(config)
        self.period = period
        self.column = column
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Linear Regression.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with Linear Regression values
        """
        result = pd.DataFrame(index=data.index)
        
        def rolling_linreg(series, window):
            """Calculate rolling linear regression."""
            slopes = []
            intercepts = []
            r_squared = []
            predictions = []
            
            for i in range(len(series)):
                if i < window - 1:
                    slopes.append(np.nan)
                    intercepts.append(np.nan)
                    r_squared.append(np.nan)
                    predictions.append(np.nan)
                else:
                    y = series.iloc[i-window+1:i+1].values
                    x = np.arange(window)
                    
                    # Calculate linear regression
                    coeffs = np.polyfit(x, y, 1)
                    slope, intercept = coeffs
                    
                    # Calculate R-squared
                    y_pred = slope * x + intercept
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    # Prediction is the last point on the regression line
                    prediction = slope * (window - 1) + intercept
                    
                    slopes.append(slope)
                    intercepts.append(intercept)
                    r_squared.append(r2)
                    predictions.append(prediction)
            
            return pd.Series(slopes), pd.Series(intercepts), pd.Series(r_squared), pd.Series(predictions)
        
        slopes, intercepts, r2, predictions = rolling_linreg(data[self.column], self.period)
        
        result[self.config.name] = predictions
        result[f"{self.config.name}_slope"] = slopes
        result[f"{self.config.name}_r2"] = r2
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return self.period


class IchimokuCloud(CompositeIndicator):
    """
    Ichimoku Cloud indicator.
    
    Comprehensive trend-following system with multiple components.
    """
    
    def __init__(self, tenkan_period: int = 9, kijun_period: int = 26, 
                 senkou_b_period: int = 52, displacement: int = 26, name: str = None):
        """
        Initialize Ichimoku Cloud.
        
        Args:
            tenkan_period: Tenkan-sen period
            kijun_period: Kijun-sen period
            senkou_b_period: Senkou Span B period
            displacement: Cloud displacement
            name: Custom name prefix for the indicator
        """
        if name is None:
            name = "ichimoku"
        
        config = IndicatorConfig(
            name=name,
            indicator_type="ICHIMOKU",
            parameters={
                'tenkan_period': tenkan_period,
                'kijun_period': kijun_period,
                'senkou_b_period': senkou_b_period,
                'displacement': displacement
            },
            input_columns=['high', 'low', 'close'],
            output_columns=[
                f"{name}_tenkan", f"{name}_kijun", f"{name}_chikou",
                f"{name}_senkou_a", f"{name}_senkou_b"
            ],
            min_periods=senkou_b_period + displacement
        )
        super().__init__(config)
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with Ichimoku values
        """
        result = pd.DataFrame(index=data.index)
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=self.tenkan_period).max()
        tenkan_low = low.rolling(window=self.tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = high.rolling(window=self.kijun_period).max()
        kijun_low = low.rolling(window=self.kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-self.displacement)
        
        # Senkou Span A (Leading Span A)
        senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(self.displacement)
        
        # Senkou Span B (Leading Span B)
        senkou_b_high = high.rolling(window=self.senkou_b_period).max()
        senkou_b_low = low.rolling(window=self.senkou_b_period).min()
        senkou_b = ((senkou_b_high + senkou_b_low) / 2).shift(self.displacement)
        
        result[f"{self.config.name}_tenkan"] = tenkan_sen
        result[f"{self.config.name}_kijun"] = kijun_sen
        result[f"{self.config.name}_chikou"] = chikou_span
        result[f"{self.config.name}_senkou_a"] = senkou_a
        result[f"{self.config.name}_senkou_b"] = senkou_b
        
        return result
    
    def get_required_periods(self) -> int:
        """Get minimum periods required."""
        return self.senkou_b_period + self.displacement 