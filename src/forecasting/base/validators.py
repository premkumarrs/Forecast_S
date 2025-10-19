"""
Base validation functions for forecasting data.
"""

import pandas as pd
from typing import Tuple, Optional


def validate_time_series(
    ts_data: pd.DataFrame,
    required_columns: Optional[list] = None,
    min_length: int = 3
) -> Tuple[bool, str]:
    """
    Validate time series data for forecasting.
    
    Args:
        ts_data: Time series DataFrame
        required_columns: List of required column names
        min_length: Minimum required length of time series
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if ts_data is None or ts_data.empty:
        return False, "Time series data is empty or None"
    
    # Check required columns
    if required_columns is None:
        required_columns = ['year', 'value']
    
    missing_columns = set(required_columns) - set(ts_data.columns)
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    # Check minimum length
    if len(ts_data) < min_length:
        return False, f"Time series too short: {len(ts_data)} < {min_length}"
    
    # Check for duplicate years
    if 'year' in ts_data.columns:
        if ts_data['year'].duplicated().any():
            return False, "Duplicate years found in time series"
    
    # Check for non-numeric values
    if 'value' in ts_data.columns:
        if not pd.api.types.is_numeric_dtype(ts_data['value']):
            return False, "Value column contains non-numeric data"
        
        # Check for NaN values
        if ts_data['value'].isna().any():
            return False, "Value column contains NaN values"
        
        # Check for negative values (optional based on context)
        if (ts_data['value'] < 0).any():
            return False, "Value column contains negative values"
    
    return True, ""


def validate_forecast_years(
    forecast_years: list,
    historical_years: list
) -> Tuple[bool, str]:
    """
    Validate forecast years against historical years.
    
    Args:
        forecast_years: List of years to forecast
        historical_years: List of historical years
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not forecast_years:
        return False, "No forecast years provided"
    
    if not historical_years:
        return False, "No historical years provided"
    
    # Check that forecast years come after historical years
    max_hist_year = max(historical_years)
    min_forecast_year = min(forecast_years)
    
    if min_forecast_year <= max_hist_year:
        return False, f"Forecast years must be after historical years (max historical: {max_hist_year})"
    
    # Check for gaps
    expected_start = max_hist_year + 1
    if min_forecast_year != expected_start:
        return False, f"Gap detected between historical and forecast years"
    
    # Check for continuity in forecast years
    sorted_years = sorted(forecast_years)
    for i in range(1, len(sorted_years)):
        if sorted_years[i] != sorted_years[i-1] + 1:
            return False, "Forecast years are not continuous"
    
    return True, ""


def validate_forecast_data(
    forecast_df: pd.DataFrame,
    required_columns: Optional[list] = None
) -> Tuple[bool, str]:
    """
    Validate forecast output data.
    
    Args:
        forecast_df: Forecast DataFrame
        required_columns: List of required column names
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if forecast_df is None or forecast_df.empty:
        return False, "Forecast data is empty or None"
    
    # Check required columns
    if required_columns is None:
        required_columns = ['year', 'value_hat', 'type']
    
    missing_columns = set(required_columns) - set(forecast_df.columns)
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    # Check type column values
    if 'type' in forecast_df.columns:
        valid_types = {'Historical', 'Forecast'}
        actual_types = set(forecast_df['type'].unique())
        invalid_types = actual_types - valid_types
        if invalid_types:
            return False, f"Invalid type values: {invalid_types}"
    
    # Check for reasonable forecast values
    if 'value_hat' in forecast_df.columns:
        forecast_values = forecast_df[forecast_df['type'] == 'Forecast']['value_hat']
        
        if forecast_values.empty:
            return False, "No forecast values found"
        
        if forecast_values.isna().any():
            return False, "Forecast contains NaN values"
        
        # Check for extreme values (optional)
        historical_values = forecast_df[forecast_df['type'] == 'Historical']['value_hat']
        if not historical_values.empty:
            hist_mean = historical_values.mean()
            hist_std = historical_values.std()
            
            # Flag if forecast is more than 10 standard deviations from historical mean
            extreme_threshold = 10
            for value in forecast_values:
                if abs(value - hist_mean) > extreme_threshold * hist_std:
                    return False, f"Extreme forecast value detected: {value}"
    
    return True, ""