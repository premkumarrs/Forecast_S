"""
Data serialization helpers for session state management.
"""

import streamlit as st
import pandas as pd
from typing import Any, Dict
import json
from datetime import datetime


def save_dataframe(key: str, df: pd.DataFrame) -> None:
    """Save a DataFrame to session state.
    
    Args:
        key (str): Session state key
        df (pd.DataFrame): DataFrame to save
    """
    # Convert DataFrame to dict for JSON serialization
    if not df.empty:
        st.session_state[key] = df.to_dict('records')
        st.session_state[f'{key}_columns'] = list(df.columns)
        st.session_state[f'{key}_index'] = list(df.index)
    else:
        st.session_state[key] = []
        st.session_state[f'{key}_columns'] = []
        st.session_state[f'{key}_index'] = []
    
    st.session_state['last_update'] = datetime.now().isoformat()


def get_dataframe(key: str) -> pd.DataFrame:
    """Get a DataFrame from session state.
    
    Args:
        key (str): Session state key
        
    Returns:
        pd.DataFrame: DataFrame from session state or empty DataFrame
    """
    data = st.session_state.get(key, [])
    columns = st.session_state.get(f'{key}_columns', [])
    index = st.session_state.get(f'{key}_index', [])
    
    if data and columns:
        df = pd.DataFrame(data, columns=columns)
        if index and len(index) == len(df):
            df.index = index
        return df
    else:
        return pd.DataFrame()


def save_forecast_results(forecasts: Dict[str, pd.Series], 
                         metadata: Dict[str, Any] = None) -> None:
    """Save forecast results to session state.
    
    Args:
        forecasts (Dict[str, pd.Series]): Forecast results by region/model
        metadata (Dict[str, Any], optional): Additional metadata
    """
    # Convert Series to dict format
    forecast_data = {}
    for key, series in forecasts.items():
        forecast_data[key] = {
            'values': series.tolist(),
            'index': series.index.tolist() if hasattr(series.index, 'tolist') else list(series.index)
        }
    
    st.session_state['forecasts'] = forecast_data
    st.session_state['forecast_metadata'] = metadata or {}
    st.session_state['forecasts_generated'] = True
    st.session_state['last_update'] = datetime.now().isoformat()


def get_forecast_results() -> Dict[str, pd.Series]:
    """Get forecast results from session state.
    
    Returns:
        Dict[str, pd.Series]: Forecast results
    """
    forecast_data = st.session_state.get('forecasts', {})
    forecasts = {}
    
    for key, data in forecast_data.items():
        if 'values' in data and 'index' in data:
            forecasts[key] = pd.Series(data['values'], index=data['index'])
    
    return forecasts


def get_forecast_metadata() -> Dict[str, Any]:
    """Get forecast metadata from session state.
    
    Returns:
        Dict[str, Any]: Forecast metadata
    """
    return st.session_state.get('forecast_metadata', {})


def export_session_state() -> Dict[str, Any]:
    """Export session state to a dictionary.
    
    Returns:
        Dict[str, Any]: Session state data
    """
    # Create a copy of session state, excluding non-serializable items
    exportable_state = {}
    
    for key, value in st.session_state.items():
        try:
            # Test if value is JSON serializable
            json.dumps(value)
            exportable_state[key] = value
        except (TypeError, ValueError):
            # Skip non-serializable values
            continue
    
    return exportable_state


def import_session_state(state_data: Dict[str, Any]) -> None:
    """Import session state from a dictionary.
    
    Args:
        state_data (Dict[str, Any]): Session state data to import
    """
    for key, value in state_data.items():
        st.session_state[key] = value
    
    st.session_state['last_update'] = datetime.now().isoformat()