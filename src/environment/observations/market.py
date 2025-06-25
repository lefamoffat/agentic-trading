#!/usr/bin/env python3
"""Market data observation component for trading environment.

This module handles market-related observations like OHLCV data.
"""
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

class MarketObservation:
    """Handles market data observations with proper normalization.
    
    This provides dynamic market feature observations.
    """
    
    def __init__(self, features: List[str], normalization_method: str = "robust_zscore",
                 window: int = 50):
        """Initialize market observation component.
        
        Args:
            features: List of market features to include
            normalization_method: How to normalize the data
            window: Lookback window for normalization
        """
        self.features = features
        self.normalization_method = normalization_method
        self.window = window
        
        # Initialize scalers for each feature
        self.scalers = {}
        if normalization_method == "robust_zscore":
            for feature in features:
                self.scalers[feature] = RobustScaler()
        
        self.reset()
    
    def reset(self) -> None:
        """Reset observation state."""
        self._fitted_scalers = False
        # Re-initialize scalers
        if self.normalization_method == "robust_zscore":
            for feature in self.features:
                self.scalers[feature] = RobustScaler()
    
    def _fit_scalers(self, data: pd.DataFrame) -> None:
        """Fit scalers on available data.
        
        Args:
            data: Market data DataFrame
        """
        if self._fitted_scalers:
            return
        
        # Use first window of data to fit scalers
        fit_data = data.head(min(self.window, len(data)))
        
        for feature in self.features:
            if feature in fit_data.columns:
                # Convert to numeric first, coerce errors to NaN
                feature_series = pd.to_numeric(fit_data[feature], errors='coerce')
                feature_data = feature_series.values.reshape(-1, 1)
                
                # Remove any NaN values for fitting - now safe since data is numeric
                valid_mask = ~np.isnan(feature_data.flatten())
                clean_data = feature_data[valid_mask].reshape(-1, 1)
                
                if len(clean_data) > 0:
                    self.scalers[feature].fit(clean_data)
        
        self._fitted_scalers = True
    
    def get_observation(self, data: pd.DataFrame, current_step: int) -> np.ndarray:
        """Get normalized market observation for current step.
        
        Args:
            data: Full market data DataFrame
            current_step: Current step index
            
        Returns:
            Normalized market features array
            
        Raises:
            ValueError: If current_step is invalid
        """
        if current_step < 0 or current_step >= len(data):
            raise ValueError(f"Invalid step {current_step} for data length {len(data)}")
        
        # Fit scalers if not done yet
        if not self._fitted_scalers:
            self._fit_scalers(data)
        
        # Get current row
        current_row = data.iloc[current_step]
        
        # Extract and normalize features
        observations = []
        for feature in self.features:
            if feature in current_row:
                value = current_row[feature]
                
                # Convert to numeric first, handle non-numeric data
                if pd.isna(value):
                    numeric_value = 0.0
                else:
                    try:
                        numeric_value = float(pd.to_numeric(value, errors='coerce'))
                        if pd.isna(numeric_value):
                            numeric_value = 0.0
                    except (ValueError, TypeError):
                        numeric_value = 0.0
                
                # Normalize if scaler is available and fitted
                if feature in self.scalers and self._fitted_scalers:
                    try:
                        normalized_value = self.scalers[feature].transform([[numeric_value]])[0, 0]
                        # Handle any extreme values
                        normalized_value = np.clip(normalized_value, -10, 10)
                    except:
                        normalized_value = 0.0
                else:
                    normalized_value = numeric_value
                
                observations.append(normalized_value)
            else:
                # Feature not found, use 0
                observations.append(0.0)
        
        return np.array(observations, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names in observation order.
        
        Returns:
            List of feature names
        """
        return self.features.copy()
    
    @property
    def observation_size(self) -> int:
        """Get size of observation vector.
        
        Returns:
            Number of features in observation
        """
        return len(self.features) 