"""
Forecasting methods module - clear separation of each forecasting approach.
"""

from .global_forecast import forecast_global_only
from .top_down import forecast_top_down
from .bottom_up import forecast_bottom_up
from .country_specific import forecast_country_specific

__all__ = [
    'forecast_global_only',
    'forecast_top_down', 
    'forecast_bottom_up',
    'forecast_country_specific'
]