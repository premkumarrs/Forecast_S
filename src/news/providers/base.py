"""
Base class for news providers.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional, Tuple


class NewsProvider(ABC):
    """Abstract base class for news data providers."""
    
    def __init__(self, name: str):
        """
        Initialize news provider.
        
        Args:
            name: Provider name
        """
        self.name = name
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
            country_code: Country code (ISO3 or provider-specific)
            topics: List of topics to search
            days_back: Number of days to look back
            progress_callback: Optional progress callback
            
        Returns:
            Articles DataFrame
        """
        pass
    
    @abstractmethod
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
            country_mapping: Mapping of country names to codes
            topics: List of topics to search
            days_back: Number of days to look back
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary mapping country names to article DataFrames
        """
        pass
    
    def validate_topics(self, topics: List[str]) -> List[str]:
        """
        Validate and clean topics.
        
        Args:
            topics: Raw topic list
            
        Returns:
            Cleaned topic list
        """
        if not topics:
            return []
        
        cleaned = []
        for topic in topics:
            if topic and isinstance(topic, str):
                topic = topic.strip()
                if topic:
                    cleaned.append(topic)
        
        return cleaned
    
    def get_provider_info(self) -> Dict:
        """
        Get provider information.
        
        Returns:
            Dictionary with provider details
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__
        }