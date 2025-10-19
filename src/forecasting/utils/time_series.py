"""
Time series utility functions.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict


def validate_time_series(ts_data: pd.DataFrame) -> bool:
    """
    Validates time series data for forecasting.
    
    Args:
        ts_data: Time series DataFrame with 'year' and 'value' columns
        
    Returns:
        bool: True if valid, False otherwise
    """
    if ts_data.empty:
        return False
    
    required_columns = ['year', 'value']
    if not all(col in ts_data.columns for col in required_columns):
        return False
    
    # Check for minimum data points
    if len(ts_data) < 3:
        return False
    
    # Check for non-null values
    if ts_data[required_columns].isnull().any().any():
        return False
    
    return True


def clean_forecast_data(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and validates forecast data.
    
    Args:
        forecast_df: DataFrame with forecast results
        
    Returns:
        Cleaned DataFrame
    """
    if forecast_df.empty:
        return forecast_df
    
    # Remove any infinite or NaN values
    forecast_df = forecast_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Ensure proper data types
    if 'year' in forecast_df.columns:
        forecast_df['year'] = forecast_df['year'].astype(int)
    
    if 'value_hat' in forecast_df.columns:
        forecast_df['value_hat'] = pd.to_numeric(
            forecast_df['value_hat'],
            errors='coerce'
        )
    
    return forecast_df


def calculate_forecast_metrics(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame
) -> Dict:
    """
    Calculate basic metrics for forecast evaluation.
    
    Args:
        historical_df: Historical data
        forecast_df: Forecast data
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    if not historical_df.empty and len(historical_df) > 1:
        # Calculate historical growth rate
        hist_values = historical_df['value'].values
        hist_growth_rates = np.diff(hist_values) / hist_values[:-1] * 100
        
        metrics['historical_avg_growth'] = np.mean(hist_growth_rates)
        metrics['historical_std_growth'] = np.std(hist_growth_rates)
        metrics['last_historical_value'] = hist_values[-1]
    
    if not forecast_df.empty and len(forecast_df) > 1:
        # Calculate forecast growth rate
        forecast_values = forecast_df['value_hat'].values
        forecast_growth_rates = (
            np.diff(forecast_values) / forecast_values[:-1] * 100
        )
        
        metrics['forecast_avg_growth'] = np.mean(forecast_growth_rates)
        metrics['forecast_std_growth'] = np.std(forecast_growth_rates)
        metrics['forecast_total_change'] = (
            (forecast_values[-1] / forecast_values[0]) - 1
        ) * 100
    
    return metrics


def fill_missing_years(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    value_col: str = 'value'
) -> pd.DataFrame:
    """
    Fill missing years in time series data.
    
    Args:
        df: DataFrame with year column
        start_year: Start year
        end_year: End year
        value_col: Name of value column
        
    Returns:
        DataFrame with all years filled
    """
    if df.empty:
        return df
    
    # Create complete year range
    all_years = pd.DataFrame({'year': range(start_year, end_year + 1)})
    
    # Merge with existing data
    filled_df = all_years.merge(df, on='year', how='left')
    
    # Interpolate missing values
    if value_col in filled_df.columns:
        filled_df[value_col] = filled_df[value_col].interpolate(method='linear')
    
    return filled_df