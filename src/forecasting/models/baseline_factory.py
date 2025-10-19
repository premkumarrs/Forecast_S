"""
Factory for creating baseline forecasting models.
"""

import pandas as pd
from typing import Dict, Type
from ..base.models import BaseForecaster
from .cagr import CAGRForecaster
from .ets import DampedETSForecaster
from .logistic import LogisticGrowthForecaster


class BaselineModelFactory:
    """Factory class for creating baseline forecasting models."""
    
    # Model registry mapping names to classes
    _models: Dict[str, Type[BaseForecaster]] = {
        "3-yr CAGR": CAGRForecaster,
        "CAGR": CAGRForecaster,
        "Damped ETS": DampedETSForecaster,
        "ETS": DampedETSForecaster,
        "Logistic Growth": LogisticGrowthForecaster,
        "Logistic": LogisticGrowthForecaster,
    }
    
    @classmethod
    def create_model(cls, model_type: str) -> BaseForecaster:
        """
        Create a forecasting model instance.
        
        Args:
            model_type: Type of model to create
            
        Returns:
            Instance of the requested forecasting model
            
        Raises:
            ValueError: If model_type is not recognized
        """
        if model_type not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available models: {available}"
            )
        
        model_class = cls._models[model_type]
        
        # Special handling for CAGR with 3-year lookback
        if model_type == "3-yr CAGR":
            return model_class(lookback_years=3)
        
        return model_class()
    
    @classmethod
    def generate_baseline_forecast(
        cls,
        global_ts: pd.DataFrame,
        model_type: str,
        hist_cutoff: int,
        forecast_until: int
    ) -> pd.DataFrame:
        """
        Generate baseline forecast using specified model.
        This method provides backward compatibility with the old API.
        
        Args:
            global_ts: Historical time series data
            model_type: Type of baseline model to use
            hist_cutoff: Last year of historical data to include
            forecast_until: Year to forecast up to
            
        Returns:
            DataFrame with columns ['year', 'value_hat', 'type']
        """
        # Create the appropriate model
        model = cls.create_model(model_type)
        
        # Generate forecast
        return model.generate_forecast_dataframe(
            global_ts,
            hist_cutoff,
            forecast_until
        )
    
    @classmethod
    def get_available_models(cls) -> list:
        """
        Get list of available model types.
        
        Returns:
            List of available model type names
        """
        return list(cls._models.keys())
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseForecaster]):
        """
        Register a new model type.
        
        Args:
            name: Name for the model type
            model_class: Class implementing BaseForecaster
        """
        if not issubclass(model_class, BaseForecaster):
            raise TypeError(
                f"{model_class} must be a subclass of BaseForecaster"
            )
        cls._models[name] = model_class