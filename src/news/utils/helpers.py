"""
Helper utility functions for news processing.
"""

import pandas as pd
from datetime import datetime


def filter_recent_news(news_df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    """
    Filter news to recent articles only.
    
    Args:
        news_df: DataFrame with news data
        days: Number of days to look back
        
    Returns:
        Filtered DataFrame with recent news
    """
    if news_df.empty or 'date' not in news_df.columns:
        return news_df
    
    cutoff_date = datetime.now() - pd.Timedelta(days=days)
    return news_df[news_df['date'] >= cutoff_date]


def deduplicate_headlines(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate headlines.
    
    Args:
        news_df: DataFrame with news data
        
    Returns:
        DataFrame without duplicate headlines
    """
    if news_df.empty or 'title' not in news_df.columns:
        return news_df
    
    return news_df.drop_duplicates(subset=['title'], keep='first')