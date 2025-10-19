"""
Page-specific helper functions.
"""

from .insights import (
    render_global_insights,
    render_regional_insights,
    render_country_insights,
    render_country_comparison,
    render_news_analysis,
    render_forecast_analysis,
    render_yoy_growth_analysis,
    create_adjustment_breakdown_table,
    prepare_country_timeline,
    get_country_news_data
)

from .forecasting import (
    calculate_impact_timelines,
    should_show_insights_page,
    get_forecast_method_display,
    get_service_method_name,
    create_forecast_adjustment_table
)

from .configuration import (
    generate_topics_with_llm,
    get_available_countries_from_session,
    validate_configuration
)

from .data_extraction import (
    get_cache_key,
    execute_unified_extraction
)

from .export import (
    build_market_wide_table,
    build_news_export
)

__all__ = [
    # Insights
    'render_global_insights',
    'render_regional_insights', 
    'render_country_insights',
    'render_country_comparison',
    'render_news_analysis',
    'render_forecast_analysis',
    'render_yoy_growth_analysis',
    'create_adjustment_breakdown_table',
    'prepare_country_timeline',
    'get_country_news_data',
    
    # Forecasting
    'calculate_impact_timelines',
    'should_show_insights_page',
    'get_forecast_method_display',
    'get_service_method_name',
    'create_forecast_adjustment_table',
    
    # Configuration
    'generate_topics_with_llm',
    'get_available_countries_from_session',
    'validate_configuration',
    
    # Data Extraction
    'get_cache_key',
    'execute_unified_extraction',
    
    # Export (minimal)
    'build_market_wide_table',
    'build_news_export'
]
