"""
News utility functions (kept lean – only used helpers exposed).
"""

from .helpers import filter_recent_news, deduplicate_headlines

__all__ = [
    'filter_recent_news',
    'deduplicate_headlines'
]
