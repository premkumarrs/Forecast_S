"""
Compound Annual Growth Rate (CAGR) forecasting model.
"""

import pandas as pd
from typing import List
from ..base.models import BaseForecaster


class CAGRForecaster(BaseForecaster):
    """Forecast based on compound annual growth rate."""
    
    def __init__(self, lookback_years: int = 3):
        """
        Initialize CAGR forecaster.
        
        Args:
            lookback_years: Number of years to use for CAGR calculation
        """
        super().__init__("CAGR")
        self.lookback_years = lookback_years
        self.cagr = None
        self.last_value = None
    
    def fit(self, historical_data: pd.DataFrame) -> None:
        """
        Calculate CAGR from historical data.
        
        Args:
            historical_data: DataFrame with 'year' and 'value' columns
        """
        # Use last N years for CAGR calculation
        last_n_years = historical_data.tail(self.lookback_years + 1)
        
        if len(last_n_years) < 2:
            self.cagr = 0.05  # Default growth rate
        else:
            start_val = last_n_years.iloc[0]['value']
            end_val = last_n_years.iloc[-1]['value']
            periods = len(last_n_years) - 1
            
            if start_val > 0:
                self.cagr = (end_val / start_val) ** (1 / periods) - 1
            else:
                self.cagr = 0.05  # Default if start value is invalid
        
        self.last_value = historical_data.iloc[-1]['value']
        self._is_fitted = True
    
    def forecast(self, forecast_years: List[int]) -> List[float]:
        """
        Generate forecast using CAGR.
        
        Args:
            forecast_years: List of years to forecast
            
        Returns:
            List of forecasted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        return [
            self.last_value * ((1 + self.cagr) ** (i + 1))
            for i in range(len(forecast_years))
        ]
    
    def get_params(self) -> dict:
        """Get model parameters including CAGR."""
        params = super().get_params()
        params.update({
            'lookback_years': self.lookback_years,
            'cagr': self.cagr,
            'last_value': self.last_value
        })
        return params