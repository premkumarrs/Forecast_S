"""
Helper functions for Forecasting page.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Tuple, Optional, List


def calculate_impact_timelines(hist_cutoff: int, forecast_until: int, 
                              unified_data: Optional[Dict] = None) -> Dict:
    """
    Calculate impact timelines for news and indicators.
    
    Args:
        hist_cutoff: Historical cutoff year
        forecast_until: Forecast end year
        unified_data: Optional unified data dictionary
        
    Returns:
        Dictionary with timeline information
    """
    # Calculate forecast timeline
    forecast_years = list(range(hist_cutoff + 1, forecast_until + 1))
    
    # Get indicator availability timeline if unified_data provided
    indicator_years = []
    if unified_data:
        indicators = unified_data.get('indicators', {})
        country_indicators = indicators.get('country', pd.DataFrame())
        if not country_indicators.empty and 'year' in country_indicators.columns:
            indicator_years = sorted(country_indicators['year'].unique().tolist())
    
    return {
        'forecast_years': forecast_years,
        'forecast_horizon': len(forecast_years),
        'indicator_years': indicator_years,
        'news_window': 30,  # Default 30-day news window
        'temporal_decay': True
    }


def should_show_insights_page() -> bool:
    """
    Check if Insights page should be shown.
    
    Returns:
        True if forecast results are available
    """
    return 'forecast_result' in st.session_state and st.session_state['forecast_result'] is not None


def get_forecast_method_display() -> str:
    """
    Get display name for selected forecast method.
    
    Returns:
        Formatted method name
    """
    config = st.session_state.get('config', {})
    method = config.get('forecast_method', '3-yr CAGR')
    
    # Add emoji based on method
    method_icons = {
        '3-yr CAGR': '📈',
        'Damped ETS': '📊',
        'Logistic Growth': '📉'
    }
    
    icon = method_icons.get(method, '📊')
    return f"{icon} {method}"


def get_service_method_name(ui_method_name: str) -> str:
    """
    Convert UI method name to service method name.
    
    Args:
        ui_method_name: Method name from UI
        
    Returns:
        Service-compatible method name
    """
    method_mapping = {
        'Global-level': 'Global Only',
        'Top-down': 'Top-Down',
        'Bottom-up': 'Bottom-Up',
        'Country-Specific': 'Country-Specific'
    }
    return method_mapping.get(ui_method_name, ui_method_name)


def create_forecast_adjustment_table(forecast_result: Dict, 
                                    analyzed_news: pd.DataFrame,
                                    config: Dict,
                                    unified_data: Dict) -> pd.DataFrame:
    """
    Create adjustment breakdown table for forecast.
    
    Args:
        forecast_result: Forecast results
        analyzed_news: Analyzed news data
        config: Configuration dictionary
        unified_data: Unified data dictionary
        
    Returns:
        DataFrame with adjustment breakdown
    """
    adjustments_data = []
    
    # Check what adjustments were applied
    adjustments_applied = forecast_result.get('adjustments_applied', [])
    
    # News adjustments
    if 'news' in adjustments_applied or 'country_news' in adjustments_applied:
        news_count = len(analyzed_news) if analyzed_news is not None else 0
        relevant_count = analyzed_news['relevant'].sum() if analyzed_news is not None and 'relevant' in analyzed_news.columns else 0
        
        adjustments_data.append({
            'Adjustment Type': '📰 News Analysis',
            'Status': '✅ Applied',
            'Details': f"{relevant_count}/{news_count} relevant articles",
            'Weight': f"{config.get('adjustment_weights', {}).get('news', 0.3) * 100:.0f}%"
        })
    else:
        adjustments_data.append({
            'Adjustment Type': '📰 News Analysis',
            'Status': '❌ Not Applied',
            'Details': 'No news data available',
            'Weight': '-'
        })
    
    # Indicator adjustments
    indicators = unified_data.get('indicators', {})
    has_indicators = not indicators.get('country', pd.DataFrame()).empty or not indicators.get('global', pd.DataFrame()).empty
    
    if 'indicators' in adjustments_applied and has_indicators:
        indicator_count = len(config.get('indicator_weights', {}))
        adjustments_data.append({
            'Adjustment Type': '📊 Economic Indicators',
            'Status': '✅ Applied',
            'Details': f"{indicator_count} indicators configured",
            'Weight': f"{config.get('adjustment_weights', {}).get('indicators', 0.7) * 100:.0f}%"
        })
    else:
        adjustments_data.append({
            'Adjustment Type': '📊 Economic Indicators',
            'Status': '❌ Not Applied',
            'Details': 'No indicators configured or available',
            'Weight': '-'
        })
    
    # Temporal decay
    if config.get('use_temporal_decay', True):
        adjustments_data.append({
            'Adjustment Type': '⏰ Temporal Decay',
            'Status': '✅ Applied',
            'Details': 'Exponential decay for news impact',
            'Weight': 'Variable'
        })
    
    # Regional fallback
    metadata = forecast_result.get('metadata', {})
    if 'fallback_countries' in metadata and metadata['fallback_countries']:
        fallback_count = metadata.get('countries_with_fallback', 0)
        adjustments_data.append({
            'Adjustment Type': '🌍 Regional Fallback',
            'Status': '✅ Applied',
            'Details': f"{fallback_count} countries used regional news",
            'Weight': 'Confidence-weighted'
        })
    
    return pd.DataFrame(adjustments_data)


def validate_forecast_requirements(config: Dict, unified_data: Dict) -> Tuple[bool, List[str]]:
    """
    Validate requirements for forecasting.
    
    Args:
        config: Configuration dictionary
        unified_data: Unified data dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check market data
    market_data = unified_data.get('market_value', {})
    if not market_data:
        errors.append("No market data available")
    
    # Check forecast approach
    if not config.get('forecast_approach'):
        errors.append("No forecast approach selected")
    
    # Check method-specific requirements
    approaches = config.get('forecast_approach', [])
    
    if 'Country-Specific' in approaches:
        if not config.get('selected_countries'):
            errors.append("Country-Specific approach requires selected countries")
    
    # Check adjustment configuration
    if config.get('use_news') and not config.get('topics'):
        errors.append("News analysis enabled but no topics configured")
    
    if config.get('use_indicators') and not config.get('indicator_weights'):
        errors.append("Indicators enabled but no weights configured")
    
    return len(errors) == 0, errors


def get_forecast_summary(forecast_result: Dict) -> Dict:
    """
    Get summary statistics from forecast result.
    
    Args:
        forecast_result: Forecast results
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'method': forecast_result.get('method', 'Unknown'),
        'adjustments_applied': forecast_result.get('adjustments_applied', []),
        'countries_forecasted': 0,
        'regions_forecasted': 0,
        'has_global': False
    }
    
    # Count entities
    if 'country_forecasts' in forecast_result:
        summary['countries_forecasted'] = len(forecast_result['country_forecasts'])
    
    if 'region_forecasts' in forecast_result:
        summary['regions_forecasted'] = len(forecast_result['region_forecasts'])
    
    if 'global_forecast' in forecast_result:
        summary['has_global'] = not forecast_result['global_forecast'].empty
    
    # Get metadata
    metadata = forecast_result.get('metadata', {})
    summary.update({
        'news_coverage': metadata.get('news_coverage', 'N/A'),
        'fallback_used': metadata.get('countries_with_fallback', 0) > 0
    })
    
    return summary