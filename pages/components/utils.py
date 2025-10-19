"""
Shared utility functions for Streamlit pages.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any


def safe_get_value_column(df: pd.DataFrame) -> Optional[str]:
    """
    Safely get the value column from a DataFrame.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Column name containing values, or None if not found
    """
    if df is None or df.empty:
        return None
    
    # Priority: value > value_hat
    if 'value' in df.columns:
        return 'value'
    elif 'value_hat' in df.columns:
        return 'value_hat'
    
    # Try common variations
    for col in df.columns:
        if 'value' in col.lower():
            return col
    
    return None


def safe_get_nested(data: Any, *keys: str, default: Any = None) -> Any:
    """
    Safely get nested dictionary values.
    
    Args:
        data: The data structure to navigate
        *keys: Sequence of keys to traverse
        default: Default value if key not found
        
    Returns:
        The value at the nested location, or default if not found
    """
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key)
            if data is None:
                return default
        else:
            return default
    return data


def get_hist_cutoff() -> int:
    """
    Get the historical cutoff year consistently.
    
    Returns:
        Historical cutoff year
    """
    return st.session_state.get('saved_hist_cutoff', 
                                st.session_state.get('hist_cutoff', 2024))


def get_forecast_until() -> int:
    """
    Get the forecast until year consistently.
    
    Returns:
        Forecast end year
    """
    # Try unified_data first for most reliable source
    unified_data = st.session_state.get('unified_data', {})
    extraction_params = unified_data.get('extraction_params', {})
    if 'forecast_until' in extraction_params:
        return extraction_params['forecast_until']
    
    # Fallback to session state
    return st.session_state.get('saved_forecast_until', 
                                st.session_state.get('forecast_until', 2030))


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame columns for consistency.
    
    Args:
        df: DataFrame to normalize
        
    Returns:
        DataFrame with normalized columns
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    # Ensure 'type' column exists
    if 'type' not in df.columns:
        if 'year' in df.columns:
            # Determine type based on year vs cutoff
            hist_cutoff = get_hist_cutoff()
            df['type'] = df['year'].apply(
                lambda y: 'historical' if y <= hist_cutoff else 'forecast'
            )
    
    # Ensure value column exists
    value_col = safe_get_value_column(df)
    if value_col and value_col != 'value_hat':
        df['value_hat'] = df[value_col]
    
    return df


def get_available_countries(forecast_result: Dict) -> List[str]:
    """
    Get all available countries from forecast result.
    
    Args:
        forecast_result: Forecast result dictionary
        
    Returns:
        Sorted list of country names
    """
    country_forecasts = forecast_result.get('country_forecasts', {})
    if not country_forecasts:
        return []
    return sorted(list(country_forecasts.keys()))


def get_available_regions(forecast_result: Dict) -> List[str]:
    """
    Get all available regions from forecast result.
    
    Args:
        forecast_result: Forecast result dictionary
        
    Returns:
        List of regions with Worldwide first if present
    """
    region_forecasts = forecast_result.get('region_forecasts', {})
    if not region_forecasts:
        return []
    
    # Sort with Worldwide first if it exists
    regions = list(region_forecasts.keys())
    if 'Worldwide' in regions:
        regions.remove('Worldwide')
        regions.sort()
        regions.insert(0, 'Worldwide')
    else:
        regions.sort()
    
    return regions


def get_session_value(key: str, default: Any = None) -> Any:
    """
    Safely get value from session state.
    
    Args:
        key: Session state key
        default: Default value if key not found
        
    Returns:
        Session state value or default
    """
    return st.session_state.get(key, default)


def set_session_value(key: str, value: Any) -> None:
    """
    Safely set value in session state.
    
    Args:
        key: Session state key
        value: Value to set
    """
    st.session_state[key] = value


def has_session_value(key: str) -> bool:
    """
    Check if session state has a key.
    
    Args:
        key: Session state key
        
    Returns:
        True if key exists in session state
    """
    return key in st.session_state


def clear_session_values(*keys: str) -> None:
    """
    Clear specific keys from session state.
    
    Args:
        *keys: Keys to clear from session state
    """
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]


def format_large_number(value: float, decimals: int = 1) -> str:
    """
    Format large numbers with K, M, B suffixes.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    if pd.isna(value):
        return "N/A"
    
    abs_value = abs(value)
    sign = "-" if value < 0 else ""
    
    if abs_value >= 1e9:
        return f"{sign}{abs_value/1e9:.{decimals}f}B"
    elif abs_value >= 1e6:
        return f"{sign}{abs_value/1e6:.{decimals}f}M"
    elif abs_value >= 1e3:
        return f"{sign}{abs_value/1e3:.{decimals}f}K"
    else:
        return f"{sign}{abs_value:.{decimals}f}"


def format_value_intelligent(value: float, force_unit: str = None) -> str:
    """Format value with appropriate unit (K/M/B) based on magnitude with $ prefix.
    
    Args:
        value: The value to format
        force_unit: Force specific unit ('K', 'M', 'B', None for auto)
    
    Returns:
        Formatted string with $ prefix
    """
    if pd.isna(value) or value == 0:
        return "$0"
    
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    
    # Auto-detect best unit if not forced
    if force_unit is None:
        if abs_val >= 1e9:
            return f"${sign}{abs_val/1e9:,.1f}B"
        elif abs_val >= 1e6:
            return f"${sign}{abs_val/1e6:,.1f}M"
        elif abs_val >= 1e3:
            return f"${sign}{abs_val/1e3:,.1f}K"
        else:
            return f"${sign}{abs_val:,.0f}"
    else:
        # Use forced unit
        if force_unit == 'B':
            return f"${sign}{abs_val/1e9:,.1f}B"
        elif force_unit == 'M':
            return f"${sign}{abs_val/1e6:,.1f}M"
        elif force_unit == 'K':
            return f"${sign}{abs_val/1e3:,.1f}K"
        else:
            return f"${sign}{abs_val:,.0f}"


def detect_data_unit(values) -> str:
    """Detect the likely unit of data based on value ranges.
    
    Args:
        values: pandas Series or list/array of values
    
    Returns:
        String indicating likely unit: 'raw', 'millions', 'thousands', 'billions'
    """
    import pandas as pd
    import numpy as np
    
    if hasattr(values, 'dropna'):
        clean_values = values.dropna()
    else:
        clean_values = [v for v in values if pd.notna(v)]
    
    if len(clean_values) == 0:
        return 'raw'
    
    # Use median to avoid outlier influence
    median_val = np.median([abs(v) for v in clean_values])
    
    if median_val > 1e12:  # Likely raw values in trillions range
        return 'raw'
    elif median_val > 1e9:  # Likely stored as millions, displayed as billions
        return 'millions'
    elif median_val > 1e6:  # Could be thousands or raw millions
        return 'thousands'
    else:  # Likely already in billions or normalized
        return 'billions'


def should_convert_to_billions(values, current_unit: str = None) -> bool:
    """Determine if values should be converted to billions for display consistency.
    
    Args:
        values: pandas Series or array of values
        current_unit: Known unit if available
    
    Returns:
        Boolean indicating if conversion to billions is appropriate
    """
    if current_unit:
        return current_unit in ['raw', 'millions']
    
    detected_unit = detect_data_unit(values)
    return detected_unit in ['raw', 'millions']


def calculate_cagr(start_value: float, end_value: float, years: int) -> float:
    """
    Calculate Compound Annual Growth Rate.
    
    Args:
        start_value: Starting value
        end_value: Ending value
        years: Number of years
        
    Returns:
        CAGR as a percentage
    """
    if start_value <= 0 or end_value <= 0 or years <= 0:
        return 0.0
    
    return ((end_value / start_value) ** (1 / years) - 1) * 100