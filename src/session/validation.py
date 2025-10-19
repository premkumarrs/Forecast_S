"""
Session validation functions.
"""

import streamlit as st
from typing import Any, Dict, List


def validate_required_data(required_keys: List[str]) -> Dict[str, Any]:
    """Validate that required data is present in session state.
    
    Args:
        required_keys (List[str]): List of required session state keys
        
    Returns:
        Dict[str, Any]: Validation results
    """
    validation = {
        'is_valid': True,
        'missing_keys': [],
        'present_keys': []
    }
    
    for key in required_keys:
        if key in st.session_state and st.session_state[key] is not None:
            validation['present_keys'].append(key)
        else:
            validation['missing_keys'].append(key)
            validation['is_valid'] = False
    
    return validation


def validate_data_consistency() -> Dict[str, Any]:
    """Validate data consistency in session state.
    
    Returns:
        Dict[str, Any]: Validation results
    """
    issues = []
    warnings = []
    
    # Check year range consistency
    hist_cutoff = st.session_state.get('hist_cutoff')
    forecast_until = st.session_state.get('forecast_until')
    
    if hist_cutoff and forecast_until:
        if forecast_until < hist_cutoff:
            issues.append("Forecast until year must be >= hist_cutoff year")
    
    # Check data availability
    if st.session_state.get('data_loaded', False):
        # Validate that key datasets exist
        if 'global_ts' not in st.session_state:
            issues.append("Global time series data missing despite data_loaded=True")
        
        if 'country_ts_full' not in st.session_state:
            warnings.append("Country time series data missing")
    
    return {
        'is_valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings
    }


def validate_forecast_readiness() -> Dict[str, Any]:
    """Validate that session state is ready for forecasting.
    
    Returns:
        Dict[str, Any]: Validation results
    """
    required_for_forecast = [
        'hist_cutoff',
        'forecast_until', 
        'global_ts'
    ]
    
    validation = validate_required_data(required_for_forecast)
    
    # Additional forecast-specific checks
    if validation['is_valid']:
        global_ts = st.session_state.get('global_ts')
        if global_ts is not None:
            if hasattr(global_ts, '__len__') and len(global_ts) < 3:
                validation['is_valid'] = False
                validation['missing_keys'].append('insufficient_historical_data')
    
    return validation