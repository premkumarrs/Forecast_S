"""
Forecast metrics and evaluation functions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def calculate_forecast_metrics(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    value_col: str = 'value'
) -> Dict:
    """
    Calculate comprehensive forecast metrics.
    
    Args:
        historical_df: Historical data DataFrame
        forecast_df: Forecast data DataFrame
        value_col: Name of value column
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic statistics
    if not historical_df.empty:
        hist_values = historical_df[value_col]
        metrics['historical_mean'] = hist_values.mean()
        metrics['historical_std'] = hist_values.std()
        metrics['historical_min'] = hist_values.min()
        metrics['historical_max'] = hist_values.max()
        
        # Growth metrics
        if len(historical_df) > 1:
            metrics['historical_cagr'] = calculate_cagr(
                hist_values.iloc[0],
                hist_values.iloc[-1],
                len(hist_values) - 1
            )
    
    if not forecast_df.empty:
        # Separate forecast values
        forecast_mask = forecast_df['type'] == 'Forecast'
        forecast_values = forecast_df[forecast_mask]['value_hat']
        
        metrics['forecast_mean'] = forecast_values.mean()
        metrics['forecast_std'] = forecast_values.std()
        metrics['forecast_min'] = forecast_values.min()
        metrics['forecast_max'] = forecast_values.max()
        
        # Forecast growth
        if len(forecast_values) > 1:
            metrics['forecast_cagr'] = calculate_cagr(
                forecast_values.iloc[0],
                forecast_values.iloc[-1],
                len(forecast_values) - 1
            )
        
        # Continuity check
        if not historical_df.empty and not forecast_values.empty:
            last_historical = hist_values.iloc[-1]
            first_forecast = forecast_values.iloc[0]
            metrics['continuity_gap'] = (first_forecast - last_historical) / last_historical * 100
    
    return metrics


def calculate_cagr(start_value: float, end_value: float, periods: int) -> float:
    """
    Calculate Compound Annual Growth Rate.
    
    Args:
        start_value: Starting value
        end_value: Ending value
        periods: Number of periods
        
    Returns:
        CAGR as percentage
    """
    if start_value <= 0 or end_value <= 0 or periods <= 0:
        return 0.0
    
    return ((end_value / start_value) ** (1 / periods) - 1) * 100


def calculate_mape(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        MAPE as percentage
    """
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted series must have same length")
    
    # Filter out zeros to avoid division by zero
    mask = actual != 0
    if not mask.any():
        return np.inf
    
    actual_filtered = actual[mask]
    predicted_filtered = predicted[mask]
    
    mape = np.mean(np.abs((actual_filtered - predicted_filtered) / actual_filtered)) * 100
    
    return mape


def calculate_rmse(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Calculate Root Mean Square Error.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        RMSE value
    """
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted series must have same length")
    
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    
    return rmse


def calculate_mae(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        MAE value
    """
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted series must have same length")
    
    mae = np.mean(np.abs(actual - predicted))
    
    return mae


def calculate_forecast_accuracy(
    actual: pd.Series,
    predicted: pd.Series,
    metrics: Optional[list] = None
) -> Dict:
    """
    Calculate multiple forecast accuracy metrics.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        metrics: List of metrics to calculate (default: all)
        
    Returns:
        Dictionary of accuracy metrics
    """
    if metrics is None:
        metrics = ['mape', 'rmse', 'mae']
    
    results = {}
    
    if 'mape' in metrics:
        results['mape'] = calculate_mape(actual, predicted)
    
    if 'rmse' in metrics:
        results['rmse'] = calculate_rmse(actual, predicted)
    
    if 'mae' in metrics:
        results['mae'] = calculate_mae(actual, predicted)
    
    # Additional metrics
    if 'bias' in metrics:
        results['bias'] = np.mean(predicted - actual)
    
    if 'correlation' in metrics:
        results['correlation'] = actual.corr(predicted)
    
    return results


def evaluate_forecast_quality(
    forecast_df: pd.DataFrame,
    thresholds: Optional[Dict] = None
) -> Dict:
    """
    Evaluate overall forecast quality based on various criteria.
    
    Args:
        forecast_df: Forecast DataFrame with historical and forecast data
        thresholds: Custom thresholds for quality assessment
        
    Returns:
        Dictionary with quality assessment
    """
    if thresholds is None:
        thresholds = {
            'max_growth_rate': 0.5,  # 50% YoY
            'min_growth_rate': -0.3,  # -30% YoY
            'volatility_threshold': 0.3  # 30% coefficient of variation
        }
    
    quality = {
        'is_valid': True,
        'warnings': [],
        'metrics': {}
    }
    
    # Separate historical and forecast
    hist_mask = forecast_df['type'] == 'Historical'
    fore_mask = forecast_df['type'] == 'Forecast'
    
    hist_values = forecast_df[hist_mask]['value_hat']
    fore_values = forecast_df[fore_mask]['value_hat']
    
    # Check growth rates
    if len(fore_values) > 1:
        growth_rates = fore_values.pct_change().dropna()
        max_growth = growth_rates.max()
        min_growth = growth_rates.min()
        
        if max_growth > thresholds['max_growth_rate']:
            quality['warnings'].append(f"Excessive growth rate: {max_growth:.1%}")
            quality['is_valid'] = False
        
        if min_growth < thresholds['min_growth_rate']:
            quality['warnings'].append(f"Excessive decline rate: {min_growth:.1%}")
            quality['is_valid'] = False
        
        quality['metrics']['max_growth_rate'] = max_growth
        quality['metrics']['min_growth_rate'] = min_growth
    
    # Check volatility
    if len(fore_values) > 2:
        cv = fore_values.std() / fore_values.mean()
        if cv > thresholds['volatility_threshold']:
            quality['warnings'].append(f"High volatility: CV={cv:.2f}")
        
        quality['metrics']['coefficient_of_variation'] = cv
    
    # Check monotonicity
    is_monotonic_increasing = (fore_values.diff().dropna() >= 0).all()
    is_monotonic_decreasing = (fore_values.diff().dropna() <= 0).all()
    
    quality['metrics']['is_monotonic'] = is_monotonic_increasing or is_monotonic_decreasing
    quality['metrics']['trend_direction'] = (
        'increasing' if is_monotonic_increasing else
        'decreasing' if is_monotonic_decreasing else
        'mixed'
    )
    
    return quality