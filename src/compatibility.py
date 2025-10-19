"""
Compatibility adapter functions to maintain backward compatibility with existing pages
when using the new unified data structure.
"""
import pandas as pd
import streamlit as st
from src.session import get_from_session, save_to_session

def ensure_backward_compatibility():
    """
    Creates backward-compatible session state variables from unified data structure.
    This function should be called by pages that depend on the old structure.
    """
    unified_data = get_from_session('unified_data')
    
    if not unified_data:
        return False  # No unified data available
    
    # Extract and save global market value data (global_ts)
    if 'market_value' in unified_data:
        market_data = unified_data['market_value']
        if 'global' in market_data and not market_data['global'].empty:
            # Convert to old format: just year and value columns
            global_ts = market_data['global'][['year', 'value']].copy()
            save_to_session('global_ts_original', global_ts)
            save_to_session('global_ts', global_ts)
            
            # Save KPI metadata
            metadata = market_data.get('metadata', {})
            save_to_session('kpi_global_value', metadata.get('kpi_key', ''))
            save_to_session('kpi_global_value_name', metadata.get('kpi_name', ''))
    
    # Extract and save global indicators data 
    if 'indicators' in unified_data:
        indicator_data = unified_data['indicators']
        if 'global' in indicator_data and not indicator_data['global'].empty:
            # Keep the existing format for global indicators (drop data_level column if it exists)
            available_columns = [col for col in ['year', 'indicator_key', 'indicator_name', 'value'] if col in indicator_data['global'].columns]
            global_indicators = indicator_data['global'][available_columns].copy()
            save_to_session('global_indicators', global_indicators)
            save_to_session('kpi_global_indicators', indicator_data.get('indicator_list', []))
    
    # Extract and save country data
    if 'market_value' in unified_data:
        market_data = unified_data['market_value']
        if 'country' in market_data and not market_data['country'].empty:
            # Keep the existing format for country data (drop data_level column for backward compatibility)
            country_columns = ['country', 'iso3', 'year', 'value', 'kpiKey', 'kpi_name']
            available_columns = [col for col in country_columns if col in market_data['country'].columns]
            country_ts_full = market_data['country'][available_columns].copy()
            save_to_session('country_ts_full', country_ts_full)
            save_to_session('kpi_country_value', market_data['metadata'].get('kpi_key', ''))
            save_to_session('kpi_country_value_name', market_data['metadata'].get('kpi_name', ''))
            
            # Extract available countries
            countries = country_ts_full['country'].unique().tolist()
            save_to_session('available_countries', countries)
            save_to_session('active_countries', countries)
    
    # Extract and save country indicators
    if 'indicators' in unified_data:
        indicator_data = unified_data['indicators']
        if 'country' in indicator_data and not indicator_data['country'].empty:
            # Keep the existing format for country indicators (drop data_level column if it exists)
            country_columns = ['country', 'iso3', 'year', 'indicator_key', 'indicator_name', 'value']
            available_columns = [col for col in country_columns if col in indicator_data['country'].columns]
            country_indicators = indicator_data['country'][available_columns].copy()
            save_to_session('country_indicators', country_indicators)
            save_to_session('kpi_country_indicators', indicator_data.get('indicator_list', []))
    
    # Extract extraction parameters
    extraction_params = unified_data.get('extraction_params', {})
    if extraction_params:
        save_to_session('hist_cutoff', extraction_params.get('hist_cutoff'))
        save_to_session('forecast_until', extraction_params.get('forecast_until'))
    
    # Mark as ready for old pages
    save_to_session('page1_ready', True)
    
    return True

def check_data_availability():
    """
    Check if either unified data or old-format data is available.
    Returns tuple (has_data, is_unified_format)
    """
    # Check for unified data first
    unified_data = get_from_session('unified_data')
    
    # Check both old and new extraction completion flags
    extraction_complete = (
        get_from_session('extraction_complete', False) or 
        get_from_session('data_loaded', False)
    )
    
    if unified_data and extraction_complete:
        return True, True
    
    # Check for old format data
    has_global_ts = 'global_ts' in st.session_state and not get_from_session('global_ts').empty
    has_page1_ready = get_from_session('page1_ready', False)
    
    if has_global_ts and has_page1_ready:
        return True, False
    
    return False, False

def get_unified_market_kpi():
    """Get the market KPI key from either unified or legacy data"""
    # Try unified data first
    unified_data = get_from_session('unified_data')
    if unified_data and 'market_value' in unified_data:
        return unified_data['market_value']['metadata'].get('kpi_key', '')
    
    # Try saved unified data
    saved_market_kpi = get_from_session('saved_market_kpi', '')
    if saved_market_kpi:
        return saved_market_kpi
    
    # Fall back to legacy data
    return get_from_session('kpi_global_value', '') or get_from_session('kpi_country_value', '')

def get_unified_indicator_kpis():
    """Get the indicator KPI keys from either unified or legacy data"""
    # Try unified data first
    unified_data = get_from_session('unified_data')
    if unified_data and 'indicators' in unified_data:
        return unified_data['indicators'].get('indicator_list', [])
    
    # Try saved unified data
    saved_indicator_kpis = get_from_session('saved_indicator_kpis', [])
    if saved_indicator_kpis:
        return saved_indicator_kpis
    
    # Fall back to legacy data
    global_indicators = get_from_session('kpi_global_indicators', [])
    country_indicators = get_from_session('kpi_country_indicators', [])
    
    # Return the non-empty one, or combine if both exist
    if global_indicators and country_indicators:
        return list(set(global_indicators + country_indicators))
    elif global_indicators:
        return global_indicators
    else:
        return country_indicators