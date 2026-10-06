"""
Base abstract class for all forecasting models.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any


class BaseForecaster(ABC):
    """Abstract base class for all forecasting models."""
    
    def __init__(self, name: str):
        self.name = name
        self._is_fitted = False
    
    @abstractmethod
    def fit(self, historical_data: pd.DataFrame) -> None:
        """
        Fit the model to historical data.
        
        Args:
            historical_data: DataFrame with 'year' and 'value' columns
        """
        pass
    
    @abstractmethod
    def forecast(self, forecast_years: List[int]) -> List[float]:
        """
        Generate forecast for specified years.
        
        Args:
            forecast_years: List of years to forecast
            
        Returns:
            List of forecasted values
        """
        pass
    
    def generate_forecast_dataframe(
        self, 
        historical_data: pd.DataFrame,
        hist_cutoff: int,
        forecast_until: int
    ) -> pd.DataFrame:
        """
        Generate complete forecast DataFrame with historical and forecasted data.
        
        Args:
            historical_data: Historical time series data
            hist_cutoff: Last year of historical data to include
            forecast_until: Year to forecast up to
            
        Returns:
            DataFrame with columns ['year', 'value_hat', 'type']
        """
        if historical_data.empty or len(historical_data) < 3:
            return pd.DataFrame(columns=['year', 'value_hat', 'type'])
        
        # Filter historical data
        # Models read the series positionally (last value, trend order), so
        # the history must be in chronological order before fitting.
        hist_data = historical_data[historical_data['year'] <= hist_cutoff].sort_values('year').copy()
        forecast_years = list(range(hist_cutoff + 1, forecast_until + 1))
        
        # Fit and forecast
        self.fit(hist_data)
        forecast_values = self.forecast(forecast_years)
        
        # Combine historical and forecast data
        hist_df = hist_data[['year', 'value']].rename(columns={'value': 'value_hat'})
        hist_df['type'] = 'Historical'
        
        fore_df = pd.DataFrame({'year': forecast_years, 'value_hat': forecast_values})
        fore_df['type'] = 'Forecast'
        
        return pd.concat([hist_df, fore_df]).reset_index(drop=True)
    
    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {'name': self.name}