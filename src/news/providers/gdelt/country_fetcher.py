"""
Country-specific news fetching for GDELT provider.
"""

import logging
import pandas as pd
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from .client import GDELTClient
from .utils import clean_topics, iso3_to_fips, calculate_fallback_confidence

logger = logging.getLogger(__name__)


class CountryNewsFetcher:
    """Fetches country-specific news articles from GDELT."""
    
    def __init__(self, client: GDELTClient, max_workers: int = 1):
        """
        Initialize country news fetcher.
        
        Args:
            client: GDELT client instance
            max_workers: Maximum parallel workers for multi-country fetch
        """
        self.client = client
        self.max_workers = max_workers
        self.iso3_to_fips_map = {}
    
    def set_iso3_mapping(self, mapping: Dict[str, str]):
        """
        Set ISO3 to FIPS mapping.
        
        Args:
            mapping: Dictionary mapping ISO3 codes to FIPS codes
        """
        self.iso3_to_fips_map = mapping
    
    def fetch_single_country(
        self,
        country_iso3: str,
        topics: List[str],
        days_back: int = 90,
        progress_callback=None
    ) -> pd.DataFrame:
        """
        Fetch news for a single country.
        
        Args:
            country_iso3: ISO3 country code
            topics: List of topics to search
            days_back: Number of days to look back
            progress_callback: Optional progress callback
            
        Returns:
            Articles DataFrame
        """
        # Convert ISO3 to FIPS
        fips_code = iso3_to_fips(country_iso3, self.iso3_to_fips_map)
        if not fips_code:
            logger.warning(f"No FIPS code found for {country_iso3}")
            return pd.DataFrame()
        
        # Clean topics
        topics = clean_topics(topics)
        if not topics:
            return pd.DataFrame()
        
        # Get date bounds
        start_date, end_date = self.client.get_date_bounds(days_back)
        
        # Batch topics and fetch
        topic_batches = self.client.batch_topics(topics)
        all_articles = []
        
        for i, batch in enumerate(topic_batches):
            try:
                articles_df = self.client.search_articles(
                    keywords=batch,
                    start_date=start_date,
                    end_date=end_date,
                    country=fips_code,
                    max_records=self.client.MAX_ARTICLES_PER_BATCH
                )
                
                if not articles_df.empty:
                    all_articles.append(articles_df)
                
                if progress_callback:
                    progress_callback(i + 1, len(topic_batches))
                    
            except Exception as e:
                logger.error(f"Failed to fetch batch for {country_iso3}: {str(e)}")
        
        # Combine and deduplicate
        if all_articles:
            combined_df = pd.concat(all_articles, ignore_index=True)
            combined_df = self.client.deduplicate_by_url(combined_df)
            return combined_df
        
        return pd.DataFrame()
    
    def fetch_multiple_countries(
        self,
        country_mapping: Dict[str, str],
        topics: List[str],
        days_back: int = 90,
        progress_callback=None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch news for multiple countries in parallel.
        
        Args:
            country_mapping: Mapping of country names to ISO3 codes
            topics: List of topics to search
            days_back: Number of days to look back
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary mapping country names to article DataFrames
        """
        if not country_mapping:
            return {}
        
        # Clean topics once
        topics = clean_topics(topics)
        if not topics:
            logger.warning("No valid topics for country news fetch")
            return {}
        
        logger.info(
            f"Fetching news for {len(country_mapping)} countries "
            f"with {len(topics)} topics"
        )
        
        results = {}
        completed = 0
        total = len(country_mapping)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks for each country
            future_to_country = {
                executor.submit(
                    self.fetch_single_country,
                    iso3_code,
                    topics,
                    days_back
                ): country_name
                for country_name, iso3_code in country_mapping.items()
            }
            
            # Process completed tasks
            for future in as_completed(future_to_country):
                country_name = future_to_country[future]
                try:
                    articles_df = future.result()
                    results[country_name] = articles_df
                    
                    completed += 1
                    article_count = len(articles_df) if not articles_df.empty else 0
                    
                    if progress_callback:
                        progress_callback(
                            completed,
                            total,
                            country_name,
                            article_count
                        )
                    
                    logger.info(
                        f"Fetched {article_count} articles for {country_name}"
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to fetch news for {country_name}: {str(e)}")
                    results[country_name] = pd.DataFrame()
                    completed += 1
        
        return results
    
    def apply_fallback(
        self,
        country_name: str,
        country_articles: pd.DataFrame,
        global_articles: pd.DataFrame,
        min_threshold: int = 10
    ) -> Dict:
        """
        Apply fallback strategy for countries with insufficient news.
        
        Args:
            country_name: Country name
            country_articles: Country-specific articles
            global_articles: Global articles to use as fallback
            min_threshold: Minimum articles needed
            
        Returns:
            Dictionary with fallback data and metadata
        """
        article_count = len(country_articles) if not country_articles.empty else 0
        
        if article_count >= min_threshold:
            # Sufficient country data
            return {
                'data': country_articles,
                'confidence': 1.0,
                'strategy': 'country_specific',
                'article_count': article_count
            }
        
        # Need fallback
        confidence = calculate_fallback_confidence(article_count, min_threshold)
        
        if article_count > 0:
            # Blend country and global data
            strategy = 'blended'
            # Weight country data higher despite lower count
            country_weight = 0.7
            global_weight = 0.3
            
            # Sample global articles to match country count
            if not global_articles.empty:
                sampled_global = global_articles.sample(
                    n=min(len(global_articles), article_count * 2),
                    random_state=42
                )
                blended_data = pd.concat(
                    [country_articles, sampled_global],
                    ignore_index=True
                )
                blended_data = self.client.deduplicate_by_url(blended_data)
            else:
                blended_data = country_articles
                
            return {
                'data': blended_data,
                'confidence': confidence,
                'strategy': strategy,
                'article_count': article_count,
                'country_weight': country_weight,
                'global_weight': global_weight
            }
        else:
            # No country data, use global only
            return {
                'data': global_articles,
                'confidence': confidence * 0.5,  # Further reduce confidence
                'strategy': 'global_fallback',
                'article_count': 0
            }