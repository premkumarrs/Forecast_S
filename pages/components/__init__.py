"""
Shared components for Streamlit pages.
"""

from .utils import (
    safe_get_value_column,
    safe_get_nested,
    get_hist_cutoff,
    get_forecast_until,
    normalize_dataframe_columns,
    get_available_countries,
    get_available_regions
)

from .charts import (
    create_global_forecast_chart,
    create_regional_composition_chart,
    create_regional_forecast_chart,
    create_multi_country_chart,
    create_cagr_comparison_chart,
    create_yoy_growth_chart,
    create_country_forecast_chart,
    create_final_values_chart
)

from .metrics import (
    calculate_global_metrics,
    calculate_regional_metrics,
    calculate_country_metrics,
    calculate_growth_impact_metrics,
    create_metrics_comparison_table
)

from .layouts import (
    render_country_pills,
    get_country_flag_emoji,
    create_sidebar_header,
    create_page_header
)

__all__ = [
    # Utils
    'safe_get_value_column',
    'safe_get_nested',
    'get_hist_cutoff',
    'get_forecast_until',
    'normalize_dataframe_columns',
    'get_available_countries',
    'get_available_regions',
    
    # Charts
    'create_global_forecast_chart',
    'create_regional_composition_chart',
    'create_regional_forecast_chart',
    'create_multi_country_chart',
    'create_cagr_comparison_chart',
    'create_yoy_growth_chart',
    'create_country_forecast_chart',
    'create_final_values_chart',
    
    # Metrics
    'calculate_global_metrics',
    'calculate_regional_metrics',
    'calculate_country_metrics',
    'calculate_growth_impact_metrics',
    'create_metrics_comparison_table',
    
    # Layouts
    'render_country_pills',
    'get_country_flag_emoji',
    'create_sidebar_header',
    'create_page_header'
]