"""
GDELT API client wrapper.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple
import pandas as pd
from src.config.security import configure_requests_ssl_for_gdelt
from gdeltdoc import GdeltDoc, Filters
from .rate_limiter import GDELTRateLimiter

logger = logging.getLogger(__name__)


class GDELTClient:
    """Wrapper for GDELT API interactions."""
    
    # Configuration constants
    BATCH_SIZE = 1
    MAX_ARTICLES_PER_BATCH = 250  # API maximum
    
    def __init__(self, rate_limiter: Optional[GDELTRateLimiter] = None):
        """
        Initialize GDELT client.
        
        Args:
            rate_limiter: Optional rate limiter instance
        """
        # Ensure GDELT SSL behavior is configured per environment
        configure_requests_ssl_for_gdelt()
        self.gd = GdeltDoc()
        self.rate_limiter = rate_limiter or GDELTRateLimiter()
    
    def search_articles(
        self,
        keywords: List[str],
        start_date: str,
        end_date: str,
        country: Optional[str] = None,
        max_records: int = MAX_ARTICLES_PER_BATCH
    ) -> pd.DataFrame:
        """
        Search for articles using GDELT API.
        
        Args:
            keywords: List of keywords to search (OR logic)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            country: Optional country code (FIPS format)
            max_records: Maximum records to retrieve
            
        Returns:
            DataFrame with article data
        """
        # Apply rate limiting
        self.rate_limiter.acquire()
        
        try:
            # Build filters
            filters = Filters(
                keyword=" OR ".join(keywords),
                start_date=start_date,
                end_date=end_date,
                num_records=max_records
            )
            
            if country:
                filters.country = country
            
            # Execute search
            articles_df = self.gd.article_search(filters)
            
            if articles_df is None:
                return pd.DataFrame()
            
            # Clean and standardize columns
            return self._standardize_dataframe(articles_df)
            
        except Exception as e:
            logger.error(f"GDELT search failed: {str(e)}")
            return pd.DataFrame()
    
    def get_date_bounds(self, days_back: int = 90) -> Tuple[str, str]:
        """
        Calculate date bounds for search.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            Tuple of (start_date, end_date) in YYYY-MM-DD format
        """
        utc_now = datetime.now(timezone.utc)
        start_date = (utc_now - timedelta(days=days_back)).strftime("%Y-%m-%d")
        end_date = utc_now.strftime("%Y-%m-%d")
        return start_date, end_date
    
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize GDELT DataFrame columns.
        
        Args:
            df: Raw GDELT DataFrame
            
        Returns:
            Standardized DataFrame
        """
        if df.empty:
            return df
        
        # Ensure required columns exist
        required_columns = ['url', 'title', 'seendate']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        
        # Convert seendate to datetime if present
        if 'seendate' in df.columns and df['seendate'].notna().any():
            try:
                df['seendate'] = pd.to_datetime(df['seendate'])
            except:
                pass
        
        return df
    
    def batch_topics(self, topics: List[str], batch_size: int = BATCH_SIZE) -> List[List[str]]:
        """
        Batch topics for efficient API calls.
        
        Args:
            topics: List of topics
            batch_size: Size of each batch
            
        Returns:
            List of topic batches
        """
        if batch_size <= 0:
            return [topics]
        
        batches = []
        for i in range(0, len(topics), batch_size):
            batch = topics[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def deduplicate_by_url(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate articles by URL.
        
        Args:
            df: DataFrame with articles
            
        Returns:
            Deduplicated DataFrame
        """
        if df.empty or 'url' not in df.columns:
            return df
        
        # Remove duplicates keeping first occurrence
        return df.drop_duplicates(subset=['url'], keep='first')
    
    def get_api_stats(self) -> dict:
        """
        Get API usage statistics.
        
        Returns:
            Dictionary with API stats
        """
        return {
            'rate_limiter': self.rate_limiter.get_stats(),
            'batch_size': self.BATCH_SIZE,
            'max_articles': self.MAX_ARTICLES_PER_BATCH
        }
