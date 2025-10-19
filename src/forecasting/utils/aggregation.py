"""
Forecast aggregation utilities.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional


def aggregate_forecasts(
    forecast_dfs: List[pd.DataFrame],
    method: str = 'sum',
    weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Aggregate multiple forecasts into a single forecast.
    
    Args:
        forecast_dfs: List of forecast DataFrames
        method: Aggregation method ('sum', 'mean', 'weighted')
        weights: Optional weights for weighted aggregation
        
    Returns:
        Aggregated forecast DataFrame
    """
    if not forecast_dfs:
        return pd.DataFrame()
    
    if len(forecast_dfs) == 1:
        return forecast_dfs[0]
    
    # Align all forecasts by year
    aligned_dfs = []
    years = None
    
    for df in forecast_dfs:
        if not df.empty and 'year' in df.columns:
            if years is None:
                years = df['year'].unique()
            else:
                years = np.intersect1d(years, df['year'].unique())
            aligned_dfs.append(df)
    
    if not aligned_dfs or years is None or len(years) == 0:
        return pd.DataFrame()
    
    # Filter to common years
    filtered_dfs = []
    for df in aligned_dfs:
        filtered_df = df[df['year'].isin(years)].copy()
        filtered_dfs.append(filtered_df)
    
    # Aggregate based on method
    if method == 'sum':
        result = _aggregate_sum(filtered_dfs, years)
    elif method == 'mean':
        result = _aggregate_mean(filtered_dfs, years)
    elif method == 'weighted' and weights:
        result = _aggregate_weighted(filtered_dfs, years, weights)
    else:
        result = _aggregate_sum(filtered_dfs, years)
    
    return result


def _aggregate_sum(dfs: List[pd.DataFrame], years: np.ndarray) -> pd.DataFrame:
    """Sum aggregation of forecasts."""
    result = pd.DataFrame({'year': sorted(years)})
    result['value_hat'] = 0
    
    for df in dfs:
        year_values = df.set_index('year')['value_hat']
        for year in result['year']:
            if year in year_values.index:
                result.loc[result['year'] == year, 'value_hat'] += year_values[year]
    
    result['type'] = 'Forecast'
    return result


def _aggregate_mean(dfs: List[pd.DataFrame], years: np.ndarray) -> pd.DataFrame:
    """Mean aggregation of forecasts."""
    result = _aggregate_sum(dfs, years)
    result['value_hat'] = result['value_hat'] / len(dfs)
    return result


def _aggregate_weighted(
    dfs: List[pd.DataFrame],
    years: np.ndarray,
    weights: Dict[str, float]
) -> pd.DataFrame:
    """Weighted aggregation of forecasts."""
    result = pd.DataFrame({'year': sorted(years)})
    result['value_hat'] = 0
    total_weight = sum(weights.values())
    
    for i, df in enumerate(dfs):
        weight = list(weights.values())[i] if i < len(weights) else 1.0
        year_values = df.set_index('year')['value_hat']
        for year in result['year']:
            if year in year_values.index:
                result.loc[result['year'] == year, 'value_hat'] += (
                    year_values[year] * weight / total_weight
                )
    
    result['type'] = 'Forecast'
    return result


def combine_forecasts(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine historical and forecast data into single DataFrame.
    
    Args:
        historical_df: Historical data
        forecast_df: Forecast data
        
    Returns:
        Combined DataFrame
    """
    if historical_df.empty and forecast_df.empty:
        return pd.DataFrame()
    
    if historical_df.empty:
        return forecast_df
    
    if forecast_df.empty:
        return historical_df
    
    # Prepare historical data
    hist_df = historical_df.copy()
    if 'type' not in hist_df.columns:
        hist_df['type'] = 'Historical'
    
    # Prepare forecast data
    fore_df = forecast_df.copy()
    if 'type' not in fore_df.columns:
        fore_df['type'] = 'Forecast'
    
    # Ensure column consistency
    if 'value' in hist_df.columns and 'value_hat' not in hist_df.columns:
        hist_df['value_hat'] = hist_df['value']
    
    # Combine
    combined = pd.concat([hist_df, fore_df], ignore_index=True)
    
    # Sort by year
    if 'year' in combined.columns:
        combined = combined.sort_values('year')
    
    return combined