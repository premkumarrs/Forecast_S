"""
Forecasting module for baseline models and unified adjustments.
"""

# Import from new modular structure
from .models.baseline_factory import BaselineModelFactory

# Backward compatibility wrapper
def generate_baseline_forecast(global_ts, model_type, hist_cutoff, forecast_until):
    """Backward compatibility wrapper for baseline forecast generation."""
    return BaselineModelFactory.generate_baseline_forecast(
        global_ts, model_type, hist_cutoff, forecast_until
    )
from .utils import validate_time_series, clean_forecast_data, calculate_forecast_metrics
from .config import *

# Import from new modular adjustment structure
from .adjustments.unified import (
    calculate_unified_adjustment,
    apply_fallback_strategy
)

# Also expose new modular components
from .adjustments import (
    NewsAdjustment,
    IndicatorAdjustment,
    TemporalDecay,
    FallbackStrategy,
    AdjustmentFactory
)

# New simplified forecasting methods
from .methods import (
    forecast_global_only,
    forecast_top_down,
    forecast_bottom_up,
    forecast_country_specific
)

# Validation
from .validation import validate_method_data, get_method_requirements

# Backward compatibility for old method names
def calculate_top_down_forecast(*args, **kwargs):
    """Deprecated: Use forecast_top_down instead."""
    return forecast_top_down(*args, **kwargs)

def calculate_bottom_up_forecast(*args, **kwargs):
    """Deprecated: Use forecast_bottom_up instead."""
    return forecast_bottom_up(*args, **kwargs)

def calculate_country_specific_forecast(*args, **kwargs):
    """Deprecated: Use forecast_country_specific instead."""
    return forecast_country_specific(*args, **kwargs)

def get_temporal_summary(analyzed_news):
    """Deprecated: Simplified to return basic stats."""
    if analyzed_news.empty:
        return {}
    return {
        'total_articles': len(analyzed_news),
        'relevant_articles': len(analyzed_news[analyzed_news.get('relevant', 0) == 1]),
        'avg_growth_rate': analyzed_news.get('growth_rate', [0]).mean()
    }

__all__ = [
    # Baseline models
    'generate_baseline_forecast',
    'BaselineModelFactory',
    
    # Utils
    'validate_time_series',
    'clean_forecast_data',
    'calculate_forecast_metrics',

    # Unified adjustment (backward compatibility)
    'calculate_unified_adjustment',
    'apply_fallback_strategy',
    
    # New modular adjustments
    'NewsAdjustment',
    'IndicatorAdjustment',
    'TemporalDecay',
    'FallbackStrategy',
    'AdjustmentFactory',

    # Forecasting methods
    'forecast_global_only',
    'forecast_top_down',
    'forecast_bottom_up',
    'forecast_country_specific',

    # Validation
    'validate_method_data',
    'get_method_requirements',

    # Backward compatibility
    'calculate_top_down_forecast',
    'calculate_bottom_up_forecast',
    'calculate_country_specific_forecast',
    'get_temporal_summary'
]
