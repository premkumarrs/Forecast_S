"""
Services layer for business logic separation from UI.
"""

# Import from new modular structure for ForecastService
from .forecast.forecast_service import ForecastService

# Import from new modular structure
from .data import DataExtractionService  
from .configuration import ConfigurationService

# Also expose new modular components
from .forecast import (
    NewsAnalysisService,
    MethodRouter,
    ForecastValidationService
)

from .base import (
    BaseService,
    ServiceException,
    ValidationException,
    DataException
)

__all__ = [
    # Main services (backward compatibility)
    'ForecastService',
    'DataExtractionService',
    'ConfigurationService',
    
    # New modular components
    'NewsAnalysisService',
    'MethodRouter',
    'ForecastValidationService',
    
    # Base classes
    'BaseService',
    'ServiceException',
    'ValidationException',
    'DataException'
]
