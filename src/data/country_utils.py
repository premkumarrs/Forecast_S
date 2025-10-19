"""
Utility functions for country data processing.
"""

import pandas as pd
import json
import os
from typing import Dict, Optional, List

def extract_country_iso3_mapping(unified_data: Dict) -> Dict[str, str]:
    """
    Extract country to ISO3 mapping from unified data structure.
    
    Args:
        unified_data: The unified data dictionary from session state
    
    Returns:
        Dictionary mapping country names to ISO3 codes
    """
    country_iso3_map = {}
    
    # Try to get from market_value country data first
    if 'market_value' in unified_data and 'country' in unified_data['market_value']:
        country_df = unified_data['market_value']['country']
        if not country_df.empty and 'country' in country_df.columns and 'iso3' in country_df.columns:
            # Get unique country-ISO3 pairs
            mapping_df = country_df[['country', 'iso3']].drop_duplicates()
            country_iso3_map = dict(zip(mapping_df['country'], mapping_df['iso3']))
    
    # Also check indicators for additional countries
    if 'indicators' in unified_data and 'country' in unified_data['indicators']:
        indicator_df = unified_data['indicators']['country']
        if not indicator_df.empty and 'country' in indicator_df.columns and 'iso3' in indicator_df.columns:
            mapping_df = indicator_df[['country', 'iso3']].drop_duplicates()
            # Merge with existing map
            for country, iso3 in zip(mapping_df['country'], mapping_df['iso3']):
                if country not in country_iso3_map:
                    country_iso3_map[country] = iso3
    
    return country_iso3_map

def get_countries_with_market_data(unified_data: Dict) -> List[str]:
    """
    Get list of countries that have MARKET DATA available (primary constraint).
    
    Args:
        unified_data: The unified data dictionary
    
    Returns:
        List of country names that have market data
    """
    countries = set()
    
    # Only check market_value data - this is the constraint
    if 'market_value' in unified_data and 'country' in unified_data['market_value']:
        country_df = unified_data['market_value']['country']
        if not country_df.empty and 'country' in country_df.columns:
            countries.update(country_df['country'].unique())
    
    return sorted(list(countries))

def get_countries_with_data(unified_data: Dict) -> List[str]:
    """
    Get list of countries that have data in the unified structure.
    
    Args:
        unified_data: The unified data dictionary
    
    Returns:
        List of country names
    """
    countries = set()
    
    # Check market_value data
    if 'market_value' in unified_data and 'country' in unified_data['market_value']:
        country_df = unified_data['market_value']['country']
        if not country_df.empty and 'country' in country_df.columns:
            countries.update(country_df['country'].unique())
    
    # Check indicators data
    if 'indicators' in unified_data and 'country' in unified_data['indicators']:
        indicator_df = unified_data['indicators']['country']
        if not indicator_df.empty and 'country' in indicator_df.columns:
            countries.update(indicator_df['country'].unique())
    
    return sorted(list(countries))

def filter_excluded_regions(countries: List[str]) -> List[str]:
    """
    Filter out regional aggregates from country list using exclusions.json.
    
    Args:
        countries: List of country/region names
    
    Returns:
        List with only actual countries (regions removed)
    """
    # Load excluded regions from config file
    try:
        exclusions_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'exclusions.json')
        with open(exclusions_path, 'r') as f:
            exclusions_data = json.load(f)
        excluded_regions = set(exclusions_data.get('excluded_regions', []))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load exclusions.json: {e}")
        # Fallback to basic exclusions
        excluded_regions = {'Global', 'World', 'Rest of World', 'Other'}
    
    return [c for c in countries if c not in excluded_regions]

def get_indicator_column_name(unified_data: Dict) -> str:
    """
    Detect the correct column name for indicators ('indicator_key' or 'indicator').
    
    Args:
        unified_data: The unified data dictionary
    
    Returns:
        Column name to use for indicators
    """
    if 'indicators' in unified_data:
        for data_type in ['country', 'global']:
            if data_type in unified_data['indicators']:
                df = unified_data['indicators'][data_type]
                if not df.empty:
                    if 'indicator_key' in df.columns:
                        return 'indicator_key'
                    elif 'indicator' in df.columns:
                        return 'indicator'
    
    # Default fallback
    return 'indicator_key'