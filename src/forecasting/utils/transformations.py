"""
Forecast transformation utilities.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


def apply_growth_constraints(
    forecast_df: pd.DataFrame,
    min_growth: float = -0.5,
    max_growth: float = 1.0
) -> pd.DataFrame:
    """
    Apply growth rate constraints to forecast.
    
    Args:
        forecast_df: Forecast DataFrame
        min_growth: Minimum allowed YoY growth rate
        max_growth: Maximum allowed YoY growth rate
        
    Returns:
        Constrained forecast DataFrame
    """
    if forecast_df.empty or len(forecast_df) < 2:
        return forecast_df
    
    df = forecast_df.copy()
    
    if 'value_hat' not in df.columns:
        return df
    
    values = df['value_hat'].values
    
    # Apply constraints year by year
    for i in range(1, len(values)):
        growth_rate = (values[i] - values[i-1]) / values[i-1]
        
        if growth_rate < min_growth:
            values[i] = values[i-1] * (1 + min_growth)
        elif growth_rate > max_growth:
            values[i] = values[i-1] * (1 + max_growth)
    
    df['value_hat'] = values
    return df


def smooth_forecast(
    forecast_df: pd.DataFrame,
    window: int = 3,
    method: str = 'rolling_mean'
) -> pd.DataFrame:
    """
    Smooth forecast values.
    
    Args:
        forecast_df: Forecast DataFrame
        window: Smoothing window size
        method: Smoothing method ('rolling_mean', 'ewm')
        
    Returns:
        Smoothed forecast DataFrame
    """
    if forecast_df.empty or 'value_hat' not in forecast_df.columns:
        return forecast_df
    
    df = forecast_df.copy()
    
    if method == 'rolling_mean':
        df['value_hat'] = df['value_hat'].rolling(
            window=window,
            min_periods=1,
            center=True
        ).mean()
    elif method == 'ewm':
        df['value_hat'] = df['value_hat'].ewm(
            span=window,
            adjust=False
        ).mean()
    
    return df


def apply_confidence_bounds(
    forecast_df: pd.DataFrame,
    confidence_level: float = 0.95,
    historical_std: Optional[float] = None
) -> pd.DataFrame:
    """
    Add confidence bounds to forecast.
    
    Args:
        forecast_df: Forecast DataFrame
        confidence_level: Confidence level (0-1)
        historical_std: Historical standard deviation
        
    Returns:
        Forecast with confidence bounds
    """
    if forecast_df.empty or 'value_hat' not in forecast_df.columns:
        return forecast_df
    
    df = forecast_df.copy()
    
    # Calculate bounds based on confidence level
    if historical_std is None:
        # Estimate from forecast variability
        if len(df) > 1:
            values = df['value_hat'].values
            growth_rates = np.diff(values) / values[:-1]
            historical_std = np.std(growth_rates)
        else:
            historical_std = 0.1  # Default 10% std
    
    # Calculate z-score for confidence level
    from scipy import stats
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Apply increasing uncertainty over time
    forecast_years = df[df['type'] == 'Forecast'] if 'type' in df.columns else df
    
    for i, (idx, row) in enumerate(forecast_years.iterrows()):
        # Uncertainty increases with forecast horizon
        uncertainty = historical_std * np.sqrt(i + 1)
        margin = z_score * uncertainty * row['value_hat']
        
        df.loc[idx, 'lower_bound'] = row['value_hat'] - margin
        df.loc[idx, 'upper_bound'] = row['value_hat'] + margin
    
    return df


def normalize_forecast(
    forecast_df: pd.DataFrame,
    base_year: Optional[int] = None,
    base_value: float = 100
) -> pd.DataFrame:
    """
    Normalize forecast to base year/value.
    
    Args:
        forecast_df: Forecast DataFrame
        base_year: Year to use as base (default: first year)
        base_value: Value to normalize to
        
    Returns:
        Normalized forecast DataFrame
    """
    if forecast_df.empty or 'value_hat' not in forecast_df.columns:
        return forecast_df
    
    df = forecast_df.copy()
    
    # Determine base year
    if base_year is None:
        base_year = df['year'].min() if 'year' in df.columns else 0
    
    # Get base year value
    if 'year' in df.columns:
        base_row = df[df['year'] == base_year]
        if not base_row.empty:
            original_base = base_row['value_hat'].iloc[0]
            if original_base != 0:
                df['value_hat'] = df['value_hat'] / original_base * base_value
    
    return df