"""
Exponential Smoothing (ETS) forecasting model with damped trend.
"""

import pandas as pd
from typing import List
from statsmodels.tsa.api import ExponentialSmoothing
from ..base.models import BaseForecaster


class DampedETSForecaster(BaseForecaster):
    """Forecast using damped additive trend ETS model."""
    
    def __init__(self):
        """Initialize Damped ETS forecaster."""
        super().__init__("Damped ETS")
        self.model = None
        self.fitted_model = None
    
    def fit(self, historical_data: pd.DataFrame) -> None:
        """
        Fit ETS model to historical data.
        
        Args:
            historical_data: DataFrame with 'year' and 'value' columns
        """
        # Create ETS model with damped additive trend
        self.model = ExponentialSmoothing(
            historical_data['value'],
            trend='add',
            damped_trend=True,
            initialization_method='estimated'
        )
        
        # Fit the model
        self.fitted_model = self.model.fit()
        self._is_fitted = True
    
    def forecast(self, forecast_years: List[int]) -> List[float]:
        """
        Generate forecast using fitted ETS model.
        
        Args:
            forecast_years: List of years to forecast
            
        Returns:
            List of forecasted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        # Generate forecast for the specified number of periods
        forecast_values = self.fitted_model.forecast(len(forecast_years))
        
        return forecast_values.tolist()
    
    def get_params(self) -> dict:
        """Get model parameters."""
        params = super().get_params()
        if self.fitted_model:
            params.update({
                'alpha': self.fitted_model.params.get('smoothing_level'),
                'beta': self.fitted_model.params.get('smoothing_trend'),
                'phi': self.fitted_model.params.get('damping_trend'),
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic
            })
        return params