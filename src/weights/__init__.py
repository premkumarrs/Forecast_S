"""
Weights module - simplified and consolidated for better maintainability.
"""

# Import everything from the consolidated weights module
from .weights import (
    calculate_regional_weights, validate_regional_weights, combine_regional_weights,
    calculate_temporal_weights, calculate_recency_weights,
    dynamic_weight_adjustment, get_confidence_weights,
    apply_weights
)

# Backward compatibility implementations
def calculate_volatility_weights(*args, **kwargs):
    return calculate_temporal_weights(*args, **kwargs)

def calculate_trend_strength_weights(*args, **kwargs):
    return calculate_temporal_weights(*args, **kwargs)

def apply_temporal_weights(*args, **kwargs):
    return apply_weights(*args, **kwargs)

def calculate_performance_weights(*args, **kwargs):
    return dynamic_weight_adjustment(*args, **kwargs)

def adaptive_weight_update(*args, **kwargs):
    return dynamic_weight_adjustment(*args, **kwargs)

# Deprecated WeightEngine class for backward compatibility
class WeightEngine:
    """Deprecated: Use functions directly instead."""
    
    def __init__(self):
        self.weights = {}
        self.weight_history = []
        self.default_weight = 1.0
    
    def calculate_regional_weights(self, regional_data, method='variance'):
        return calculate_regional_weights(regional_data, method)
    
    def calculate_temporal_weights(self, data, decay_factor=0.95):
        return calculate_temporal_weights(data, decay_factor)
    
    def apply_weights(self, forecasts, weights):
        return apply_weights(forecasts, weights, self.default_weight)
    
    def dynamic_weight_adjustment(self, forecasts, actual, learning_rate=0.1):
        return dynamic_weight_adjustment(forecasts, actual, learning_rate)
    
    def get_confidence_weights(self, forecasts, confidence_scores):
        return get_confidence_weights(forecasts, confidence_scores)

__all__ = [
    # Core simplified functions
    'calculate_regional_weights', 'validate_regional_weights', 'combine_regional_weights',
    'calculate_temporal_weights', 'calculate_recency_weights',
    'dynamic_weight_adjustment', 'get_confidence_weights',
    'apply_weights',
    
    # Backward compatibility
    'WeightEngine',
    'calculate_volatility_weights', 'calculate_trend_strength_weights', 'apply_temporal_weights',
    'calculate_performance_weights', 'adaptive_weight_update'
]