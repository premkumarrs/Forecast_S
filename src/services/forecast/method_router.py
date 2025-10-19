"""
Method routing and mapping for forecast service.
"""

import logging
from typing import Dict, List, Optional, Callable
from ...forecasting.methods import (
    forecast_global_only,
    forecast_top_down,
    forecast_bottom_up,
    forecast_country_specific
)

logger = logging.getLogger(__name__)


class MethodRouter:
    """Routes forecasting methods to their implementations."""
    
    # Clear mapping of UI method names to functions
    METHOD_FUNCTIONS: Dict[str, Callable] = {
        'Global Only': forecast_global_only,
        'Top-Down': forecast_top_down,
        'Bottom-Up': forecast_bottom_up,
        'Country-Specific': forecast_country_specific
    }
    
    # Method name aliases for normalization
    METHOD_ALIASES = {
        'global only': 'Global Only',
        'global level': 'Global Only',
        'global': 'Global Only',
        'top down': 'Top-Down',
        'topdown': 'Top-Down',
        'top-down': 'Top-Down',
        'bottom up': 'Bottom-Up',
        'bottomup': 'Bottom-Up',
        'bottom-up': 'Bottom-Up',
        'country specific': 'Country-Specific',
        'countryspecific': 'Country-Specific',
        'country-specific': 'Country-Specific'
    }
    
    @classmethod
    def get_method_function(cls, method: str) -> Optional[Callable]:
        """
        Get the function for a given method name.
        
        Args:
            method: Method name (can be aliased)
            
        Returns:
            Method function or None if not found
        """
        # Normalize method name
        normalized = cls.normalize_method_name(method)
        return cls.METHOD_FUNCTIONS.get(normalized)
    
    @classmethod
    def normalize_method_name(cls, method: str) -> str:
        """
        Normalize method name to canonical form.
        
        Args:
            method: Raw method name
            
        Returns:
            Canonical method name
        """
        # Try direct match first
        if method in cls.METHOD_FUNCTIONS:
            return method
        
        # Try normalized version
        normalized = method.lower().replace('_', ' ').replace('-', ' ').strip()
        
        # Check aliases
        if normalized in cls.METHOD_ALIASES:
            return cls.METHOD_ALIASES[normalized]
        
        # Return original if no match
        return method
    
    @classmethod
    def get_all_methods(cls) -> List[str]:
        """Get list of all available method names."""
        return list(cls.METHOD_FUNCTIONS.keys())
    
    @classmethod
    def is_valid_method(cls, method: str) -> bool:
        """Check if a method name is valid."""
        return cls.get_method_function(method) is not None
    
    @classmethod
    def get_method_requirements(cls, method: str) -> Dict:
        """
        Get requirements for a specific method.
        
        Args:
            method: Method name
            
        Returns:
            Dictionary describing method requirements
        """
        normalized = cls.normalize_method_name(method)
        
        requirements = {
            'Global Only': {
                'requires_global_data': True,
                'requires_country_data': False,
                'requires_country_selection': False,
                'news_type': 'global',
                'description': 'Forecasts at global level only using global market data and news'
            },
            'Top-Down': {
                'requires_global_data': True,
                'requires_country_data': True,
                'requires_country_selection': False,
                'news_type': 'combined',
                'description': 'Forecasts global then distributes to countries using market shares'
            },
            'Bottom-Up': {
                'requires_global_data': False,
                'requires_country_data': True,
                'requires_country_selection': False,
                'news_type': 'country',
                'description': 'Forecasts each country individually then aggregates to global'
            },
            'Country-Specific': {
                'requires_global_data': False,
                'requires_country_data': True,
                'requires_country_selection': True,
                'news_type': 'country',
                'description': 'Forecasts only selected countries individually'
            }
        }
        
        return requirements.get(normalized, {})
    
    @classmethod
    def get_news_routing(cls, method: str) -> str:
        """
        Determine news fetching strategy for a method.
        
        Args:
            method: Method name
            
        Returns:
            News routing strategy ('global', 'country', 'combined')
        """
        normalized = cls.normalize_method_name(method)
        
        routing = {
            'Global Only': 'global',
            'Top-Down': 'combined',
            'Bottom-Up': 'country',
            'Country-Specific': 'country'
        }
        
        return routing.get(normalized, 'global')