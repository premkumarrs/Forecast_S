"""
Consolidated weights module combining regional, temporal, and dynamic weight calculations.
Simplified from multiple files into essential functions only.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


# Regional Weight Calculations
def calculate_regional_weights(regional_data: Dict[str, pd.Series], method: str = 'equal') -> Dict[str, float]:
    """Calculate weights for different regions (simplified)."""
    if not regional_data:
        return {}
    
    num_regions = len(regional_data)
    
    if method == 'variance':
        # Inverse variance weighting
        variances = {region: max(data.var(), 0.01) for region, data in regional_data.items()}
        inv_variances = {region: 1/var for region, var in variances.items()}
        total = sum(inv_variances.values())
        return {region: weight/total for region, weight in inv_variances.items()}
    
    elif method == 'volume':
        # Volume-based weighting
        volumes = {region: data.mean() for region, data in regional_data.items()}
        total = sum(volumes.values())
        return {region: vol/total for region, vol in volumes.items()} if total > 0 else {}
    
    else:  # Equal weighting (default)
        return {region: 1/num_regions for region in regional_data.keys()}


# Temporal Weight Calculations  
def calculate_temporal_weights(data: pd.Series, decay_factor: float = 0.95) -> pd.Series:
    """Calculate temporal weights with exponential decay."""
    if data.empty:
        return pd.Series()
    
    # More recent data gets higher weight
    weights = [decay_factor ** i for i in range(len(data)-1, -1, -1)]
    weight_series = pd.Series(weights, index=data.index)
    
    # Normalize
    return weight_series / weight_series.sum()


def calculate_recency_weights(data: pd.Series, half_life_periods: int = 5) -> pd.Series:
    """Calculate recency weights using half-life decay."""
    if data.empty:
        return pd.Series()
    
    decay_rate = 0.5 ** (1 / half_life_periods)
    return calculate_temporal_weights(data, decay_rate)


# Dynamic Weight Calculations
def dynamic_weight_adjustment(forecasts: Dict[str, pd.Series], actual: pd.Series, learning_rate: float = 0.1) -> Dict[str, float]:
    """Dynamically adjust weights based on forecast performance (simplified)."""
    if not forecasts or actual.empty:
        return {}
    
    # Calculate simple MAE for each forecast
    errors = {}
    for region, forecast in forecasts.items():
        if len(forecast) > 0 and len(actual) > 0:
            # Use overlapping periods only
            common_idx = forecast.index.intersection(actual.index)
            if len(common_idx) > 0:
                mae = abs(forecast.loc[common_idx] - actual.loc[common_idx]).mean()
                errors[region] = mae
    
    if not errors:
        # Equal weights if no valid comparisons
        return {region: 1/len(forecasts) for region in forecasts.keys()}
    
    # Inverse error weighting
    inv_errors = {region: 1/(error + 0.001) for region, error in errors.items()}
    total = sum(inv_errors.values())
    
    return {region: weight/total for region, weight in inv_errors.items()}


def get_confidence_weights(forecasts: Dict[str, pd.Series], confidence_scores: Dict[str, float]) -> Dict[str, float]:
    """Calculate weights based on model confidence scores."""
    if not forecasts or not confidence_scores:
        return {}
    
    # Normalize confidence scores
    total_confidence = sum(confidence_scores.values())
    if total_confidence <= 0:
        # Equal weights if no confidence info
        return {region: 1/len(forecasts) for region in forecasts.keys()}
    
    return {region: score/total_confidence for region, score in confidence_scores.items()}


# Utility Functions
def apply_weights(forecasts: Dict[str, pd.Series], weights: Dict[str, float], default_weight: float = 1.0) -> pd.Series:
    """Apply weights to combine multiple forecasts."""
    if not forecasts or not weights:
        return pd.Series()
        
    min_length = min(len(forecast) for forecast in forecasts.values())
    
    weighted_forecast = None
    total_weight = 0
    
    for region, forecast in forecasts.items():
        weight = weights.get(region, default_weight)
        
        if weighted_forecast is None:
            weighted_forecast = forecast.iloc[:min_length] * weight
        else:
            weighted_forecast += forecast.iloc[:min_length] * weight
            
        total_weight += weight
    
    if total_weight > 0:
        weighted_forecast = weighted_forecast / total_weight
        
    return weighted_forecast


def validate_regional_weights(weights: Dict[str, float]) -> bool:
    """Validate that regional weights are properly normalized."""
    if not weights:
        return False
    
    total = sum(weights.values())
    return abs(total - 1.0) < 0.01  # Allow small floating point errors


def combine_regional_weights(weights1: Dict[str, float], weights2: Dict[str, float], alpha: float = 0.5) -> Dict[str, float]:
    """Combine two sets of regional weights."""
    if not weights1 and not weights2:
        return {}
    elif not weights1:
        return weights2.copy()
    elif not weights2:
        return weights1.copy()
    
    combined = {}
    all_regions = set(weights1.keys()) | set(weights2.keys())
    
    for region in all_regions:
        w1 = weights1.get(region, 0)
        w2 = weights2.get(region, 0)
        combined[region] = alpha * w1 + (1 - alpha) * w2
    
    # Normalize
    total = sum(combined.values())
    if total > 0:
        combined = {region: weight/total for region, weight in combined.items()}
    
    return combined