"""
Configuration constants for forecasting operations.
"""

# Default forecasting parameters
DEFAULT_GROWTH_RATE = 0.05  # 5% default growth rate
MIN_HISTORICAL_YEARS = 3    # Minimum years of historical data required

# Model configuration
SUPPORTED_MODELS = [
    '3-yr CAGR',
    'Damped ETS', 
    'Logistic Growth'
]

# Adjustment parameters
DEFAULT_SENSITIVITY = 0.1   # Default sensitivity for indicator adjustments
MAX_ADJUSTMENT_FACTOR = 2.0 # Maximum multiplicative adjustment factor

# News aggregation methods
NEWS_AGGREGATION_METHODS = [
    'weighted_mean',
    'mean',
    'median',
    'max_abs'
]

# Recency parameters
DEFAULT_RECENCY_HALF_LIFE = 30  # Days for news recency weighting

# Logistic growth model bounds
LOGISTIC_MAX_ITERATIONS = 5000
LOGISTIC_MIN_K = 0.01
LOGISTIC_MAX_K = 5.0