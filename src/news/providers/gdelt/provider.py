"""
Main GDELT provider implementation.
"""

import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple
from ..base import NewsProvider
from .client import GDELTClient
from .rate_limiter import GDELTRateLimiter
from .global_fetcher import GlobalNewsFetcher
from .country_fetcher import CountryNewsFetcher
from .utils import load_iso3_fips_mapping

logger = logging.getLogger(__name__)


class GDELTProvider(NewsProvider):
    """GDELT news provider implementation."""
    
    def __init__(
        self,
        min_rate_limit: float = 5.0,
        max_rate_limit: float = 6.0,
        max_workers: int = 1
    ):
        """
        Initialize GDELT provider.
        
        Args:
            min_rate_limit: Minimum seconds between API calls
            max_rate_limit: Maximum seconds between API calls
            max_workers: Maximum parallel workers
        """
        super().__init__("GDELT")
        
        # Initialize components
        self.rate_limiter = GDELTRateLimiter(min_rate_limit, max_rate_limit)
        self.client = GDELTClient(self.rate_limiter)
        self.global_fetcher = GlobalNewsFetcher(self.client, max_workers)
        self.country_fetcher = CountryNewsFetcher(self.client, max_workers)
        
        # Load ISO3 to FIPS mapping
        self.iso3_to_fips = load_iso3_fips_mapping()
        self.country_fetcher.set_iso3_mapping(self.iso3_to_fips)
    
    def fetch_global_news(
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
        return self.global_fetcher.fetch(topics, days_back, progress_callback)
    
    def fetch_country_news(
        self,
        country_code: str,
        topics: List[str],
        days_back: int = 90,
        progress_callback=None
    ) -> pd.DataFrame:
        """
        Fetch news for a specific country.
        
        Args:
            country_code: ISO3 country code
            topics: List of topics to search
            days_back: Number of days to look back
            progress_callback: Optional progress callback
            
        Returns:
            Articles DataFrame
        """
        return self.country_fetcher.fetch_single_country(
            country_code,
            topics,
            days_back,
            progress_callback
        )
    
    def fetch_multi_country_news(
        self,
        country_mapping: Dict[str, str],
        topics: List[str],
        days_back: int = 90,
        progress_callback=None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch news for multiple countries.
        
        Args:
            country_mapping: Mapping of country names to ISO3 codes
            topics: List of topics to search
            days_back: Number of days to look back
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary mapping country names to article DataFrames
        """
        return self.country_fetcher.fetch_multiple_countries(
            country_mapping,
            topics,
            days_back,
            progress_callback
        )
    
    def apply_country_fallback(
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
            global_articles: Global articles for fallback
            min_threshold: Minimum articles needed
            
        Returns:
            Dictionary with fallback data and metadata
        """
        return self.country_fetcher.apply_fallback(
            country_name,
            country_articles,
            global_articles,
            min_threshold
        )
    
    def get_provider_info(self) -> Dict:
        """
        Get provider information and statistics.
        
        Returns:
            Dictionary with provider details
        """
        info = super().get_provider_info()
        info.update({
            'api_stats': self.client.get_api_stats(),
            'iso3_mappings': len(self.iso3_to_fips)
        })
        return info


# Backward compatibility functions
def fetch_max_gdelt_articles(
    topics: List[str],
    days_back: int = 90,
    progress_callback=None
) -> Tuple[pd.DataFrame, bool]:
    """
    Backward compatibility wrapper for global news fetch.
    
    Args:
        topics: List of topics to search
        days_back: Number of days to look back
        progress_callback: Optional progress callback
        
    Returns:
        Tuple of (articles DataFrame, success flag)
    """
    provider = GDELTProvider()
    return provider.fetch_global_news(topics, days_back, progress_callback)


def fetch_country_headlines(
    country_iso3: str,
    topics: List[str],
    days_back: int = 90,
    progress_callback=None
) -> pd.DataFrame:
    """
    Backward compatibility wrapper for country news fetch.
    
    Args:
        country_iso3: ISO3 country code
        topics: List of topics to search
        days_back: Number of days to look back
        progress_callback: Optional progress callback
        
    Returns:
        Articles DataFrame
    """
    provider = GDELTProvider()
    return provider.fetch_country_news(
        country_iso3,
        topics,
        days_back,
        progress_callback
    )


def fetch_multi_country_headlines(
    country_iso3_map: Dict[str, str],
    topics: List[str],
    days_back: int = 90,
    progress_callback=None
) -> Dict[str, pd.DataFrame]:
    """
    Backward compatibility wrapper for multi-country news fetch.
    
    Args:
        country_iso3_map: Mapping of country names to ISO3 codes
        topics: List of topics to search
        days_back: Number of days to look back
        progress_callback: Optional progress callback
        
    Returns:
        Dictionary mapping country names to article DataFrames
    """
    provider = GDELTProvider()
    return provider.fetch_multi_country_news(
        country_iso3_map,
        topics,
        days_back,
        progress_callback
    )


def apply_country_fallback(
    country_name: str,
    country_headlines: pd.DataFrame,
    global_headlines: pd.DataFrame,
    headlines_threshold: int = 10,
    fallback_confidence: float = 0.5
) -> Tuple[pd.DataFrame, float, str]:
    """
    Backward compatibility wrapper for fallback strategy.
    
    Args:
        country_name: Country name
        country_headlines: Country-specific articles
        global_headlines: Global articles
        headlines_threshold: Minimum articles needed
        fallback_confidence: Default confidence for fallback
        
    Returns:
        Tuple of (articles DataFrame, confidence, strategy description)
    """
    provider = GDELTProvider()
    result = provider.apply_country_fallback(
        country_name,
        country_headlines,
        global_headlines,
        headlines_threshold
    )
    
    # Convert to old format
    strategy_desc = f"{result['strategy']} (confidence: {result['confidence']:.0%})"
    return result['data'], result['confidence'], strategy_desc