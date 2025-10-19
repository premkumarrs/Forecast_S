"""
Unified adjustment functions for backward compatibility.
This module maintains the original API while using the new modular structure.
"""

import pandas as pd
from typing import Dict, Optional, Tuple
from .adjustment_factory import AdjustmentFactory
from .fallback_strategies import FallbackStrategy


def calculate_unified_adjustment(
    baseline_df: pd.DataFrame,
    news_data: Optional[pd.DataFrame],
    indicators_df: Optional[pd.DataFrame],
    config: Dict,
    news_confidence_multiplier: float = 1.0
) -> Tuple[pd.DataFrame, Dict]:
    """
    Unified adjustment function with weighted balance and proper decay.
    
    This function maintains backward compatibility with the original API
    while using the new modular adjustment system.
    
    Args:
        baseline_df: Baseline forecast DataFrame
        news_data: News data for adjustment
        indicators_df: Indicator data for adjustment
        config: Configuration parameters
        news_confidence_multiplier: Confidence in news data
        
    Returns:
        Tuple of (adjusted DataFrame, adjustment details)
    """
    # Extract weights - check both streamlit session and config structure
    weights = config.get('adjustment_weights', {})
    
    # Try streamlit session first if available
    try:
        import streamlit as st
        if hasattr(st, 'session_state'):
            session_weights = st.session_state.get('adjustment_weights', {})
            if session_weights:
                weights = session_weights
    except:
        pass
    
    # Update config with confidence multiplier
    config = config.copy()
    config['news_confidence_multiplier'] = news_confidence_multiplier
    config['adjustment_weights'] = weights
    
    # Auto-detect which adjustments to use based on data availability
    if news_data is not None and not news_data.empty:
        config['use_news'] = True
    if indicators_df is not None and not indicators_df.empty:
        config['use_indicators'] = True
    
    # Create factory and apply adjustments
    factory = AdjustmentFactory()
    adjusted_df, details = factory.apply_all_adjustments(
        baseline_df, news_data, indicators_df, config
    )
    
    # Transform details to match original format
    formatted_details = _format_details_for_compatibility(details, config)
    
    return adjusted_df, formatted_details


def apply_fallback_strategy(
    country: str,
    global_news: pd.DataFrame,
    config: Dict
) -> Tuple[pd.DataFrame, float, str]:
    """
    Apply fallback when country lacks data.
    
    This function maintains backward compatibility with the original API
    while using the new fallback strategy system.
    
    Args:
        country: Country name
        global_news: Global news data
        config: Configuration parameters
        
    Returns:
        Tuple of (news data, confidence multiplier, fallback reason)
    """
    strategy = FallbackStrategy()
    return strategy.apply(country, global_news, config)


def _format_details_for_compatibility(details: Dict, config: Dict) -> Dict:
    """
    Format adjustment details to match original API format.
    
    Args:
        details: Details from new adjustment system
        config: Configuration parameters
        
    Returns:
        Details formatted for backward compatibility
    """
    # Extract individual adjustment details
    news_adjustment = 0.0
    indicator_adjustment = 0.0
    year_adjustments = {}
    category_breakdown = None
    
    for adj_type, adj_details in details.get('adjustments', []):
        if adj_type == 'news':
            # Use single news adjustment value
            news_adjustment = adj_details.get('news_adjustment', 0.0)
            # Get category breakdown if available
            category_breakdown = adj_details.get('category_breakdown')
            
            # Merge year adjustments with new single-rate structure
            for year, year_data in adj_details.get('year_adjustments', {}).items():
                if year not in year_adjustments:
                    year_adjustments[year] = {}
                year_adjustments[year].update({
                    'news_pct': year_data.get('news_pct', 0.0),
                    'decay': year_data.get('decay', 1.0),
                    'news_avg_pct': year_data.get('news_avg_pct', 0.0),
                })
        
        elif adj_type == 'indicators':
            indicator_adjustment = adj_details.get('average_adjustment', 0.0)
            
            # Merge indicator adjustments
            for year, year_data in adj_details.get('year_adjustments', {}).items():
                if year not in year_adjustments:
                    year_adjustments[year] = {}
                year_adjustments[year].update({
                    'indicators_pct': year_data.get('indicator_pct', 0.0),
                    'raw_ind_pct': year_data.get('indicator_pct', 0.0) / 100
                })
    
    # Calculate total percentages for each year
    for year in year_adjustments:
        year_data = year_adjustments[year]
        year_data['total_pct'] = (
            year_data.get('news_pct', 0) +
            year_data.get('indicators_pct', 0)
        )
    
    # Extract weights
    weights = config.get('adjustment_weights', {})
    news_weight = weights.get('news_weight', 0.7)
    indicator_weight = weights.get('indicator_weight', 0.3)
    
    # Format final details with single news adjustment
    formatted = {
        'news_adjustment': news_adjustment,
        'indicator_adjustment': indicator_adjustment,
        'news_weight': news_weight,
        'indicator_weight': indicator_weight,
        'news_confidence_multiplier': config.get('news_confidence_multiplier', 1.0),
        'year_adjustments': year_adjustments,
        'total_adjustment': details.get('total_adjustment', 0.0),
        'news_contribution': news_weight * (news_adjustment / 100.0) * 100.0,  # % units
        'indicator_contribution': indicator_weight * indicator_adjustment
    }
    
    # Add category breakdown if available
    if category_breakdown:
        formatted['category_breakdown'] = category_breakdown
    
    # Add fallback info if present
    if 'fallback' in details:
        formatted['fallback_info'] = details['fallback']
    
    return formatted
