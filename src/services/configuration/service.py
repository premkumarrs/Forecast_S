"""
Main configuration service orchestrator.
"""

import logging
from typing import Dict, List, Tuple, Optional

from .validator import ConfigurationValidator
from .generator import ConfigurationGenerator
from .storage import ConfigurationStorage

logger = logging.getLogger(__name__)


class ConfigurationService:
    """Main service for handling configuration operations."""
    
    def __init__(self):
        self.validator = ConfigurationValidator()
        self.generator = ConfigurationGenerator()
        self.storage = ConfigurationStorage()
    
    def validate_configuration(self, config: Dict, unified_data: Dict) -> Tuple[bool, List[str]]:
        """Validate complete configuration."""
        return self.validator.validate_configuration(config, unified_data)
    
    def get_enabled_methods(self, config: Dict, unified_data: Dict) -> List[str]:
        """Get list of enabled forecasting methods."""
        return self.validator.get_enabled_methods(config, unified_data)
    
    def get_configuration_summary(self, config: Dict) -> Dict:
        """Generate summary of current configuration."""
        return {
            'market_name': config.get('market_name', 'Not set'),
            'forecast_method': config.get('forecast_method', 'Not set'),
            'forecast_approaches': config.get('forecast_approach', []),
            'selected_countries': config.get('selected_countries', []),
            'use_indicators': config.get('use_indicators', False),
            'use_news': config.get('use_news', False),
            'indicator_count': len(config.get('indicator_weights', {})),
            'topic_count': len(config.get('topics', [])),
            'adjustment_weights': config.get('adjustment_weights', {})
        }
    
    def generate_topics_with_llm(self, market_name: str, num_topics: int = 50) -> List[str]:
        """Generate topics using LLM."""
        return self.generator.generate_topics(market_name, num_topics)
    
    def generate_market_categories(self, market_name: str, num_categories: int = 8, 
                                  min_growth: float = -10.0, max_growth: float = 15.0) -> List[Dict]:
        """Generate categories for a market."""
        return self.generator.generate_categories(market_name, num_categories, min_growth, max_growth)
    
    def save_configuration(self, config: Dict) -> bool:
        """Save configuration."""
        return self.storage.save_configuration(config)
    
    def load_configuration(self) -> Dict:
        """Load configuration."""
        return self.storage.load_configuration()
    
    def load_market_categories(self, market_name: str) -> List[Dict]:
        """Load categories for a market."""
        return self.storage.load_market_categories(market_name)
    
    def save_market_categories(self, market_name: str, categories: List[Dict]) -> bool:
        """Save categories for a market."""
        return self.storage.save_market_categories(market_name, categories)
    
    def load_market_topics(self, market_name: str) -> List[Dict]:
        """Load topics for a market."""
        return self.storage.load_market_topics(market_name)
    
    def save_market_topics(self, market_name: str, topics: List[Dict]) -> bool:
        """Save topics for a market."""
        return self.storage.save_market_topics(market_name, topics)
    
    def get_original_forecast_methods(self) -> List[str]:
        """Get original forecast methods."""
        return ["3-yr CAGR", "Damped ETS", "Logistic Growth"]
    
    def get_original_forecast_approaches(self) -> List[str]:
        """Get original forecast approaches."""
        return ["Top-down", "Bottom-up", "Global-level", "Country-Specific"]
    
    def get_market_name_from_data(self, unified_data: Dict) -> str:
        """Extract market name from unified data metadata."""
        return self.storage.get_market_name_from_data(unified_data)
    
    def get_countries_with_market_data(self, unified_data: Dict) -> List[str]:
        """Get countries that have market data."""
        return self.storage.get_countries_with_market_data(unified_data)
    
    def get_available_indicators(self, unified_data: Dict, countries_with_market_data: Optional[List[str]] = None) -> List[str]:
        """Get list of available indicators."""
        return self.storage.get_available_indicators(unified_data, countries_with_market_data)
    
    def get_indicator_column_name(self, unified_data: Dict) -> str:
        """Detect the correct indicator column name."""
        return self.storage.get_indicator_column_name(unified_data)
    
    def configure_indicator_weights(self, unified_data: Dict, market_countries: List[str]) -> Dict:
        """Configure indicator weights for countries with market data."""
        return self.storage.configure_indicator_weights(unified_data, market_countries)
    
    def create_default_indicator_weights(self, available_indicators: List[str]) -> Dict[str, float]:
        """Create default equal weights for indicators."""
        return self.generator.create_default_indicator_weights(available_indicators)
    
    def validate_indicator_weights(self, weights: Dict[str, float]) -> Tuple[bool, str]:
        """Validate indicator weights."""
        return self.validator.validate_indicator_weights(weights)
    
    def validate_categories(self, categories: List[Dict]) -> Tuple[bool, List[str]]:
        """Validate category configuration."""
        return self.validator.validate_categories(categories)
    
    def validate_topic_configuration(self, topics: List[Dict]) -> Tuple[bool, List[str]]:
        """Validate topic configuration."""
        return self.validator.validate_topic_configuration(topics)
    
    def create_default_topics(self, market_name: str) -> List[Dict]:
        """Create default topics for a market."""
        return self.generator.create_default_topics(market_name)