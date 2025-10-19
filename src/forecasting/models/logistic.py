"""
Logistic (S-curve) growth forecasting model.
"""

import pandas as pd
import numpy as np
from typing import List
from scipy.optimize import curve_fit
from ..base.models import BaseForecaster


class LogisticGrowthForecaster(BaseForecaster):
    """Forecast using logistic (S-curve) growth model."""
    
    def __init__(self):
        """Initialize Logistic Growth forecaster."""
        super().__init__("Logistic Growth")
        self.params = None
        self.historical_length = None
        self.fallback_forecaster = None
    
    @staticmethod
    def logistic_function(x, L, k, x0):
        """
        Logistic function: L / (1 + exp(-k*(x-x0)))
        
        Args:
            x: Input values
            L: Maximum value (carrying capacity)
            k: Growth rate
            x0: Midpoint
            
        Returns:
            Logistic function values
        """
        return L / (1 + np.exp(-k * (x - x0)))
    
    def fit(self, historical_data: pd.DataFrame) -> None:
        """
        Fit logistic curve to historical data.
        
        Args:
            historical_data: DataFrame with 'year' and 'value' columns
        """
        x_data = np.arange(len(historical_data))
        y_data = historical_data['value'].values
        self.historical_length = len(historical_data)
        
        try:
            # Initial parameter estimates
            L_init = y_data.max() * 1.5  # Carrying capacity estimate
            k_init = 0.5  # Growth rate estimate
            x0_init = len(x_data) / 2  # Midpoint estimate
            
            # Set reasonable bounds for parameters
            bounds = (
                [y_data.max(), 0.01, 0],  # Lower bounds
                [y_data.max() * 10, 5.0, len(x_data) * 2]  # Upper bounds
            )
            
            # Fit the logistic curve
            self.params, _ = curve_fit(
                self.logistic_function,
                x_data,
                y_data,
                p0=[L_init, k_init, x0_init],
                bounds=bounds,
                maxfev=5000
            )
            self._is_fitted = True
            
        except (RuntimeError, ValueError) as e:
            # If logistic fit fails, prepare fallback to CAGR
            from .cagr import CAGRForecaster
            self.fallback_forecaster = CAGRForecaster()
            self.fallback_forecaster.fit(historical_data)
            self._is_fitted = True
    
    def forecast(self, forecast_years: List[int]) -> List[float]:
        """
        Generate forecast using fitted logistic model.
        
        Args:
            forecast_years: List of years to forecast
            
        Returns:
            List of forecasted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        # Use fallback if logistic fit failed
        if self.fallback_forecaster:
            return self.fallback_forecaster.forecast(forecast_years)
        
        # Generate forecast using logistic function
        future_x = np.arange(
            self.historical_length,
            self.historical_length + len(forecast_years)
        )
        
        forecast_values = self.logistic_function(future_x, *self.params)
        
        return forecast_values.tolist()
    
    def get_params(self) -> dict:
        """Get model parameters."""
        params_dict = super().get_params()
        
        if self.params is not None:
            params_dict.update({
                'L': self.params[0],  # Carrying capacity
                'k': self.params[1],  # Growth rate
                'x0': self.params[2],  # Midpoint
                'used_fallback': False
            })
        elif self.fallback_forecaster:
            params_dict.update({
                'used_fallback': True,
                'fallback_params': self.fallback_forecaster.get_params()
            })
        
        return params_dict