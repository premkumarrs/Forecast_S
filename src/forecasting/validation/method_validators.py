"""
Method-specific validation logic.
"""

from typing import Dict, List, Optional
from .data_validators import (
    validate_global_data,
    validate_top_down_data,
    validate_bottom_up_data,
    validate_country_specific_data
)


def validate_method_data(
    method: str,
    unified_data: Dict,
    selected_countries: Optional[List[str]] = None
) -> Dict:
    """
    Universal validation function for any forecasting method.
    
    Args:
        method: Forecasting method name
        unified_data: Unified data dictionary
        selected_countries: Optional list of selected countries
        
    Returns:
        Validation result dictionary
    """
    validators = {
        'Global Only': lambda: validate_global_data(unified_data),
        'Top-Down': lambda: validate_top_down_data(unified_data),
        'Bottom-Up': lambda: validate_bottom_up_data(unified_data),
        'Country-Specific': lambda: validate_country_specific_data(
            unified_data,
            selected_countries or []
        )
    }
    
    validator = validators.get(method)
    if validator:
        return validator()
    else:
        return {
            'valid': False,
            'message': f'Unknown forecasting method: {method}'
        }


def get_method_requirements(method: str) -> Dict:
    """
    Get data requirements for each method.
    
    Args:
        method: Forecasting method name
        
    Returns:
        Dictionary describing method requirements
    """
    requirements = {
        'Global Only': {
            'requires_global': True,
            'requires_countries': False,
            'min_countries': 0,
            'description': 'Requires global market data only'
        },
        'Top-Down': {
            'requires_global': True,
            'requires_countries': True,
            'min_countries': 1,
            'description': 'Requires global data + country data for distribution'
        },
        'Bottom-Up': {
            'requires_global': False,
            'requires_countries': True,
            'min_countries': 2,
            'description': 'Requires multiple countries with market data'
        },
        'Country-Specific': {
            'requires_global': False,
            'requires_countries': True,
            'min_countries': 1,
            'description': 'Requires selected countries to have market data'
        }
    }
    
    return requirements.get(method, {})