"""
Configuration storage module for persisting configurations.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ConfigurationStorage:
    """Handles configuration storage and retrieval."""
    
    def save_configuration(self, config: Dict) -> bool:
        """Save configuration to session state."""
        try:
            import streamlit as st
            st.session_state['config'] = config
            
            # Ensure adjustment_weights are accessible in session state
            if 'adjustment_weights' in config:
                st.session_state['adjustment_weights'] = config['adjustment_weights']
            
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def load_configuration(self) -> Dict:
        """Load configuration from session state."""
        try:
            import streamlit as st
            return st.session_state.get('config', {})
        except Exception:
            return {}
    
    def load_market_categories(self, market_name: str) -> List[Dict]:
        """Load categories for a market."""
        try:
            from ...llm.categories import load_market_categories
            return load_market_categories(market_name)
        except Exception as e:
            logger.error(f"Failed to load market categories: {e}")
            return []
    
    def save_market_categories(self, market_name: str, categories: List[Dict]) -> bool:
        """Save categories for a market."""
        try:
            from ...llm.categories import save_market_categories
            save_market_categories(market_name, categories)
            return True
        except Exception as e:
            logger.error(f"Failed to save market categories: {e}")
            return False
    
    def load_market_topics(self, market_name: str) -> List[Dict]:
        """Load topics for a market."""
        try:
            from ...llm.categories import load_market_topics
            return load_market_topics(market_name)
        except Exception as e:
            logger.error(f"Failed to load market topics: {e}")
            return []
    
    def save_market_topics(self, market_name: str, topics: List[Dict]) -> bool:
        """Save topics for a market."""
        try:
            from ...llm.categories import save_market_topics
            save_market_topics(market_name, topics)
            return True
        except Exception as e:
            logger.error(f"Failed to save market topics: {e}")
            return False
    
    def get_market_name_from_data(self, unified_data: Dict) -> str:
        """Extract market name from unified data metadata."""
        market_value_metadata = unified_data.get('market_value', {}).get('metadata', {})
        
        # Try kpi_name first
        market_name = market_value_metadata.get('kpi_name', '')
        
        # Fallback to kpi_key if name not available
        if not market_name or market_name == 'Global Market':
            market_name = market_value_metadata.get('kpi_key', 'Market Analysis')
        
        return market_name
    
    def get_countries_with_market_data(self, unified_data: Dict) -> List[str]:
        """Get countries that have market data."""
        market_data = unified_data.get('market_value', {}).get('country', pd.DataFrame())
        
        if market_data.empty or 'country' not in market_data.columns:
            return []
        
        return sorted(list(market_data['country'].unique()))
    
    def get_indicator_column_name(self, unified_data: Dict) -> str:
        """Detect the correct indicator column name."""
        indicator_data = unified_data.get('indicators', {}).get('country', pd.DataFrame())
        
        if indicator_data.empty:
            return 'indicator_key'
        
        # Check for possible column names
        possible_names = ['indicator_key', 'indicator', 'kpiKey', 'indicator_name']
        
        for name in possible_names:
            if name in indicator_data.columns:
                return name
        
        return 'indicator_key'
    
    def get_available_indicators(self, unified_data: Dict, countries_with_market_data: Optional[List[str]] = None) -> List[str]:
        """Get list of available indicators."""
        indicator_data = unified_data.get('indicators', {}).get('country', pd.DataFrame())
        
        if indicator_data.empty:
            return []
        
        # Filter to countries with market data if provided
        if countries_with_market_data:
            indicator_data = indicator_data[indicator_data['country'].isin(countries_with_market_data)]
        
        # Get indicator column name
        indicator_col = 'indicator_key' if 'indicator_key' in indicator_data.columns else 'indicator'
        
        if indicator_col not in indicator_data.columns:
            return []
        
        return list(indicator_data[indicator_col].unique())
    
    def configure_indicator_weights(self, unified_data: Dict, market_countries: List[str]) -> Dict:
        """Configure indicator weights for countries with market data."""
        indicator_data = unified_data.get('indicators', {}).get('country', pd.DataFrame())
        
        if indicator_data.empty or not market_countries:
            return {}
        
        # Filter indicators to only countries with market data
        filtered_data = indicator_data[indicator_data['country'].isin(market_countries)]
        
        # Get indicator column name
        indicator_col = self.get_indicator_column_name(unified_data)
        
        if indicator_col not in filtered_data.columns:
            return {}
        
        # Get available indicators
        available_indicators = sorted(filtered_data[indicator_col].unique())
        
        return {indicator: 0.5 for indicator in available_indicators}