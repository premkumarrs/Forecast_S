"""
Simplified session state management for Streamlit application.
Use st.session_state directly instead of wrapper functions.
"""

import streamlit as st
from typing import Dict, Any
from datetime import datetime

from ..constants import ForecastMode


def init_session_state():
    """Initialize session state with default values."""
    defaults = {
        'hist_cutoff': 2020,
        'forecast_until': 2025,
        'global_kpi_key': '',
        'country_kpi_key': '',
        'data_loaded': False,
        'forecasts_generated': False,
        'current_page': 'SQL Data Pull',
        'last_update': None,
        'forecast_mode': ForecastMode.CLASSIC.value,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_session_summary() -> Dict[str, Any]:
    """Get a summary of current session state."""
    return {
        'total_keys': len(st.session_state),
        'data_loaded': st.session_state.get('data_loaded', False),
        'forecasts_generated': st.session_state.get('forecasts_generated', False),
        'current_page': st.session_state.get('current_page', 'Unknown'),
        'last_update': st.session_state.get('last_update', 'Never'),
        'parameters': {
            'hist_cutoff': st.session_state.get('hist_cutoff'),
            'forecast_until': st.session_state.get('forecast_until'),
            'global_kpi_key': st.session_state.get('global_kpi_key'),
            'country_kpi_key': st.session_state.get('country_kpi_key')
        }
    }


# Backward compatibility wrapper functions (deprecated but kept for existing code)
def save_to_session(key: str, value: Any) -> None:
    """Deprecated: Use st.session_state[key] = value directly."""
    st.session_state[key] = value
    st.session_state['last_update'] = datetime.now().isoformat()


def get_from_session(key: str, default: Any = None) -> Any:
    """Deprecated: Use st.session_state.get(key, default) directly."""
    return st.session_state.get(key, default)


def clear_session_data(keys=None) -> None:
    """Deprecated: Use del st.session_state[key] directly."""
    if keys is None:
        keys_to_keep = ['current_page']
        keys_to_clear = [key for key in st.session_state.keys() if key not in keys_to_keep]
    else:
        keys_to_clear = keys
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    st.session_state['last_update'] = datetime.now().isoformat()


# Auto-initialize session state when module is imported
if 'session_initialized' not in st.session_state:
    init_session_state()
    st.session_state['session_initialized'] = True
