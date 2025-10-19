"""
Forecast adjustment modules for integrating news and indicators.
"""

from .base import BaseAdjustment
from .news_adjustment import NewsAdjustment
from .indicator_adjustment import IndicatorAdjustment
from .temporal_decay import TemporalDecay
from .fallback_strategies import FallbackStrategy
from .adjustment_factory import AdjustmentFactory

# Backward compatibility - import unified function
from .unified import calculate_unified_adjustment, apply_fallback_strategy

__all__ = [
    'BaseAdjustment',
    'NewsAdjustment',
    'IndicatorAdjustment',
    'TemporalDecay',
    'FallbackStrategy',
    'AdjustmentFactory',
    'calculate_unified_adjustment',  # Backward compatibility
    'apply_fallback_strategy'  # Backward compatibility
]