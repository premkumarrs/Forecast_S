"""
News processing module for GDELT integration and sentiment analysis.
"""

# Import from original gdelt.py for full functionality
from .gdelt import (
    fetch_max_gdelt_articles,
    fetch_country_headlines,
    fetch_multi_country_headlines,
    apply_country_fallback,
    get_country_article_summary
)

# Also expose new modular components
from .providers import NewsProvider, GDELTProvider

# Keep other imports minimal and focused on used helpers
from .utils import filter_recent_news, deduplicate_headlines

__all__ = [
    # Backward compatibility exports
    'fetch_max_gdelt_articles',
    'filter_recent_news',
    'deduplicate_headlines',
    'fetch_country_headlines',
    'fetch_multi_country_headlines', 
    'apply_country_fallback',
    'get_country_article_summary',
    
    # New modular components
    'NewsProvider',
    'GDELTProvider'
]
