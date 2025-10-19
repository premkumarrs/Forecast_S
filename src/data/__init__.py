"""
Data processing module for ingestion, validation, and preparation.
"""

# Import main functions to maintain backward compatibility
from .ingest import (
    normalize_data, clean_column_names, standardize_date_formats, handle_duplicates
)
from .validation import (
    validate_data_quality, validate_time_series_data, 
    check_data_completeness, validate_value_ranges
)
from .preparation import (
    prepare_time_series, split_train_test, create_lag_features,
    create_rolling_features, resample_time_series
)
from .transforms import (
    log_transform, difference_transform, normalize_columns,
    inverse_transform, detect_outliers
)
from .country_utils import (
    extract_country_iso3_mapping, get_countries_with_data, filter_excluded_regions
)

__all__ = [
    # Ingestion functions
    'normalize_data', 'clean_column_names', 'standardize_date_formats', 'handle_duplicates',
    
    # Validation functions
    'validate_data_quality', 'validate_time_series_data', 
    'check_data_completeness', 'validate_value_ranges',
    
    # Preparation functions
    'prepare_time_series', 'split_train_test', 'create_lag_features',
    'create_rolling_features', 'resample_time_series',
    
    # Transform functions
    'log_transform', 'difference_transform', 'normalize_columns',
    'inverse_transform', 'detect_outliers',
    
    # Country utilities
    'extract_country_iso3_mapping', 'get_countries_with_data', 'filter_excluded_regions'
]