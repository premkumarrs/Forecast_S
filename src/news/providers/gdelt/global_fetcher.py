"""
Global news fetching for GDELT provider.
"""

import logging
import pandas as pd
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from .client import GDELTClient
from .utils import clean_topics

logger = logging.getLogger(__name__)


class GlobalNewsFetcher:
    """Fetches global news articles from GDELT."""
    
    def __init__(self, client: GDELTClient, max_workers: int = 1):
        """
        Initialize global news fetcher.
        
        Args:
            client: GDELT client instance
            max_workers: Maximum parallel workers
        """
        self.client = client
        self.max_workers = max_workers
    
    def fetch(
        self,
        topics: List[str],
        days_back: int = 90,
        progress_callback=None
    ) -> Tuple[pd.DataFrame, bool]:
        """
        Fetch global news articles.
        
        Args:
            topics: List of topics to search
            days_back: Number of days to look back
            progress_callback: Optional progress callback
            
        Returns:
            Tuple of (articles DataFrame, success flag)
        """
        # Clean and validate topics
        topics = clean_topics(topics)
        if not topics:
            logger.warning("No valid topics provided for global news fetch")
            return pd.DataFrame(), False
        
        # Get date bounds
        start_date, end_date = self.client.get_date_bounds(days_back)
        
        # Batch topics for efficient API calls
        topic_batches = self.client.batch_topics(topics)
        total_batches = len(topic_batches)
        
        logger.info(
            f"Fetching global news: {len(topics)} topics in "
            f"{total_batches} batches, {days_back} days back"
        )
        
        # Fetch articles in parallel
        all_articles = []
        completed_batches = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batch tasks
            future_to_batch = {
                executor.submit(
                    self._fetch_batch,
                    batch,
                    start_date,
                    end_date
                ): i
                for i, batch in enumerate(topic_batches)
            }
            
            # Process completed tasks
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    articles_df = future.result()
                    if not articles_df.empty:
                        all_articles.append(articles_df)
                    
                    completed_batches += 1
                    
                    # Update progress
                    if progress_callback:
                        article_count = sum(len(df) for df in all_articles)
                        progress_callback(
                            completed_batches,
                            total_batches,
                            article_count,
                            None  # No country for global
                        )
                    
                except Exception as e:
                    logger.error(f"Failed to fetch batch {batch_idx}: {str(e)}")
                    completed_batches += 1
        
        # Combine and deduplicate results
        if all_articles:
            combined_df = pd.concat(all_articles, ignore_index=True)
            combined_df = self.client.deduplicate_by_url(combined_df)
            
            logger.info(f"Global news fetch complete: {len(combined_df)} unique articles")
            return combined_df, True
        else:
            logger.warning("No global news articles found")
            return pd.DataFrame(), False
    
    def _fetch_batch(
        self,
        topics: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch articles for a single topic batch.
        
        Args:
            topics: Topics in this batch
            start_date: Start date
            end_date: End date
            
        Returns:
            Articles DataFrame
        """
        return self.client.search_articles(
            keywords=topics,
            start_date=start_date,
            end_date=end_date,
            country=None,  # Global search
            max_records=self.client.MAX_ARTICLES_PER_BATCH
        )