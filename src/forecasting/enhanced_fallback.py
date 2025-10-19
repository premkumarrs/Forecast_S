"""
Enhanced fallback strategy prioritizing indicators over news when data is insufficient.
"""

import pandas as pd
from typing import Dict, Optional, Tuple
import os
from dotenv import load_dotenv

# Try multiple locations for .env file
if os.path.exists('config/.env'):
    load_dotenv('config/.env')
else:
    load_dotenv()  # Load from default location

def apply_enhanced_fallback(
    country_name: str,
    country_news: pd.DataFrame,
    country_indicators: pd.DataFrame,
    global_news: Optional[pd.DataFrame] = None,
    config: Optional[Dict] = None
) -> Dict:
    """
    Simplified fallback strategy with only 2 scenarios: sufficient data or insufficient data.
    
    Args:
        country_name: Name of the country
        country_news: Country-specific news DataFrame
        country_indicators: Country-specific indicators DataFrame
        global_news: Optional global news for fallback
        config: Optional configuration with thresholds
    
    Returns:
        Dictionary with:
        - news_to_use: DataFrame of news to use
        - news_weight: Weight for news impact (0-1)
        - indicator_boost: Multiplier for indicator sensitivity (1-2)
        - strategy: Description of strategy used
    """
    
    # Get configuration
    min_threshold = int(os.getenv('COUNTRY_MIN_ARTICLES_THRESHOLD', '10'))
    if config:
        min_threshold = config.get('min_articles_threshold', min_threshold)
    
    # Count articles
    article_count = len(country_news) if not country_news.empty else 0
    
    # Simplified 2-scenario approach
    if article_count >= min_threshold:
        # Scenario 1: Sufficient data - use country news normally
        return {
            'news_to_use': _ensure_temporal_fields(country_news),
            'news_weight': 1.0,
            'indicator_boost': 1.0,
            'strategy': 'sufficient_data',
            'description': f'{article_count} articles - sufficient data'
        }
    else:
        # Scenario 2: Insufficient data - boost indicators and use global news
        news_to_use = global_news if global_news is not None and not global_news.empty else pd.DataFrame()
        return {
            'news_to_use': _ensure_temporal_fields(news_to_use),
            'news_weight': 0.5 if not news_to_use.empty else 0.0,
            'indicator_boost': 1.5,  # Simple 50% boost
            'strategy': 'insufficient_data',
            'description': f'Only {article_count} articles - boosting indicators'
        }

def calculate_indicator_momentum(
    indicators_df: pd.DataFrame,
    weights: Dict[str, float]
) -> float:
    """
    Simplified indicator momentum calculation.
    
    Args:
        indicators_df: DataFrame with indicator data
        weights: Dictionary of indicator weights
    
    Returns:
        Simple momentum score
    """
    if indicators_df.empty or not weights:
        return 0.0
    
    # Calculate simple YoY growth for recent year
    indicator_col = 'indicator_key' if 'indicator_key' in indicators_df.columns else 'indicator'
    momentum_score = 0.0
    
    for indicator_key, weight in weights.items():
        ind_data = indicators_df[indicators_df[indicator_col] == indicator_key]
        if len(ind_data) >= 2:
            # Get simple growth rate from last two years
            values = ind_data.sort_values('year')['value'].values
            growth_rate = (values[-1] - values[-2]) / values[-2] if values[-2] != 0 else 0
            momentum_score += growth_rate * weight
    
    return momentum_score

def _ensure_temporal_fields(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure news DataFrame has required fields for adjustment functions.
    
    Args:
        news_df: DataFrame with news data
    
    Returns:
        DataFrame with growth_rate field guaranteed to exist
    """
    if news_df.empty:
        return news_df
    
    # Make a copy to avoid modifying original
    result_df = news_df.copy()
    
    # Ensure growth_rate field exists for compatibility
    if 'growth_rate' not in result_df.columns and 'impact' in result_df.columns:
        result_df['growth_rate'] = result_df['impact']
    elif 'growth_rate' not in result_df.columns:
        result_df['growth_rate'] = 0.0
    
    return result_df

def get_fallback_summary(fallback_result: Dict) -> str:
    """
    Get human-readable summary of simplified fallback strategy.
    
    Args:
        fallback_result: Result from apply_enhanced_fallback
    
    Returns:
        Summary string
    """
    strategy = fallback_result['strategy']
    indicator_boost = fallback_result['indicator_boost']
    
    if strategy == 'sufficient_data':
        return "Using country news and indicators normally"
    else:  # insufficient_data
        return f"Insufficient news - indicators boosted {indicator_boost:.1f}x"