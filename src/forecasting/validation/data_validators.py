"""
Data validation functions for forecasting methods.
"""

import pandas as pd
from typing import Dict, List, Optional


def validate_global_data(unified_data: Dict) -> Dict:
    """
    Check if global forecast can be performed.
    
    Args:
        unified_data: Unified data dictionary
        
    Returns:
        Validation result dictionary
    """
    result = {'valid': True, 'message': '', 'warnings': []}
    
    # Check global market data
    global_market = unified_data.get('market_value', {}).get('global')
    if global_market is None or global_market.empty:
        result['valid'] = False
        result['message'] = "No global market data available. Please complete Data Extraction first."
        return result
    
    # Check minimum data points
    if len(global_market) < 3:
        result['valid'] = False
        result['message'] = f"Insufficient global data points ({len(global_market)}). Need at least 3 years of data."
        return result
    
    # Check for indicators (optional warning)
    global_indicators = unified_data.get('indicators', {}).get('global')
    if global_indicators is None or global_indicators.empty:
        result['warnings'].append("No global indicators available - only baseline forecast will be generated")
    
    return result


def validate_country_data(unified_data: Dict, countries: Optional[List[str]] = None) -> Dict:
    """
    Check if country-level forecast can be performed.
    
    Args:
        unified_data: Unified data dictionary
        countries: Optional list of specific countries to validate
        
    Returns:
        Validation result dictionary
    """
    result = {
        'valid': True, 
        'message': '', 
        'countries_with_data': [], 
        'countries_insufficient_data': [],
        'warnings': []
    }
    
    country_market = unified_data.get('market_value', {}).get('country')
    if country_market is None or country_market.empty:
        result['valid'] = False
        result['message'] = "No country market data available. Please complete Data Extraction first."
        return result
    
    available_countries = country_market['country'].unique() if 'country' in country_market.columns else []
    
    if countries:
        # Check specific countries
        for country in countries:
            if country not in available_countries:
                result['countries_insufficient_data'].append(f"{country}: not found in data")
                continue
                
            country_data = country_market[country_market['country'] == country]
            if len(country_data) >= 3:
                result['countries_with_data'].append(country)
            else:
                result['countries_insufficient_data'].append(
                    f"{country}: only {len(country_data)} data points (need 3+)"
                )
    else:
        # Check all countries
        for country in available_countries:
            country_data = country_market[country_market['country'] == country]
            if len(country_data) >= 3:
                result['countries_with_data'].append(country)
            else:
                result['countries_insufficient_data'].append(
                    f"{country}: only {len(country_data)} data points"
                )
    
    # Final validation
    if not result['countries_with_data']:
        result['valid'] = False
        result['message'] = "No countries have sufficient data (need 3+ years per country)"
    else:
        # Add warnings for countries with insufficient data
        if result['countries_insufficient_data']:
            result['warnings'].append(
                f"Some countries excluded due to insufficient data: "
                f"{len(result['countries_insufficient_data'])} countries"
            )
    
    return result


def validate_top_down_data(unified_data: Dict) -> Dict:
    """
    Check if top-down forecast can be performed.
    
    Args:
        unified_data: Unified data dictionary
        
    Returns:
        Validation result dictionary
    """
    # Need both global and country data
    global_validation = validate_global_data(unified_data)
    country_validation = validate_country_data(unified_data)
    
    if not global_validation['valid']:
        return global_validation
    
    if not country_validation['valid']:
        # Can still do global-only if countries fail
        return {
            'valid': True,
            'message': '',
            'warnings': ['No country data available - will generate global forecast only'] + 
                       global_validation.get('warnings', [])
        }
    
    # Combine warnings
    combined_warnings = (
        global_validation.get('warnings', []) + 
        country_validation.get('warnings', [])
    )
    
    return {
        'valid': True,
        'message': '',
        'warnings': combined_warnings,
        'countries_with_data': country_validation['countries_with_data']
    }


def validate_bottom_up_data(unified_data: Dict) -> Dict:
    """
    Check if bottom-up forecast can be performed.
    
    Args:
        unified_data: Unified data dictionary
        
    Returns:
        Validation result dictionary
    """
    return validate_country_data(unified_data)


def validate_country_specific_data(
    unified_data: Dict,
    selected_countries: List[str]
) -> Dict:
    """
    Check if country-specific forecast can be performed.
    
    Args:
        unified_data: Unified data dictionary
        selected_countries: List of selected countries
        
    Returns:
        Validation result dictionary
    """
    if not selected_countries:
        return {
            'valid': False,
            'message': 'No countries selected for country-specific forecast'
        }
    
    return validate_country_data(unified_data, selected_countries)