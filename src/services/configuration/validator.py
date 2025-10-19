"""
Configuration validation module.
"""

import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class ConfigurationValidator:
    """Handles configuration validation."""
    
    def validate_configuration(self, config: Dict, unified_data: Dict) -> Tuple[bool, List[str]]:
        """Validate complete configuration."""
        errors = []
        
        # Basic validation
        if not config.get('market_name', '').strip():
            errors.append("Market name is required")
        
        if not config.get('forecast_approach'):
            errors.append("At least one forecast approach must be selected")

        # Method-specific validation
        selected_approaches = config.get('forecast_approach', [])
        
        if 'Country-Specific' in selected_approaches:
            selected_countries = config.get('selected_countries', [])
            if not selected_countries:
                errors.append("Country-specific approach requires at least one country to be selected")
            else:
                # Check if selected countries have data
                available_countries = self._get_available_countries(unified_data)
                missing_countries = [c for c in selected_countries if c not in available_countries]
                if missing_countries:
                    errors.append(f"Selected countries not found in data: {', '.join(missing_countries)}")
        
        if 'Bottom-up' in selected_approaches:
            available_countries = self._get_available_countries(unified_data)
            if len(available_countries) < 2:
                errors.append("Bottom-up approach requires at least 2 countries with data")
        
        # Indicator validation
        forecast_mode = config.get('forecast_mode')
        if config.get('use_indicators', False) and forecast_mode != 'existing_forecast_news':
            indicator_weights = config.get('indicator_weights', {})
            if not indicator_weights:
                errors.append("Indicators are enabled but no weights are configured")
        
        # News validation
        if config.get('use_news', False):
            topics = config.get('topics', [])
            if not topics:
                errors.append("News analysis is enabled but no topics are configured")
        
        return len(errors) == 0, errors
    
    def get_enabled_methods(self, config: Dict, unified_data: Dict) -> List[str]:
        """Get list of enabled forecasting methods based on configuration."""
        enabled_methods = []
        
        # Check what's enabled in config
        forecast_approaches = config.get('forecast_approach', [])
        
        # Map config names to method names
        method_mapping = {
            'Global-level': 'Global Only',
            'Top-down': 'Top-Down', 
            'Bottom-up': 'Bottom-Up',
            'Country-Specific': 'Country-Specific'
        }
        
        for approach in forecast_approaches:
            if approach in method_mapping:
                method_name = method_mapping[approach]
                
                # Additional validation for method availability
                if self._is_method_available(method_name, unified_data, config):
                    enabled_methods.append(method_name)
        
        # Fallback to Global Only if nothing is available
        if not enabled_methods:
            enabled_methods = ['Global Only']
        
        return enabled_methods
    
    def validate_indicator_weights(self, weights: Dict[str, float]) -> Tuple[bool, str]:
        """Validate indicator weights."""
        if not weights:
            return False, "No weights provided"
        
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
            return False, f"Weights must sum to 1.0 (current sum: {total_weight:.3f})"
        
        for indicator, weight in weights.items():
            if weight < 0 or weight > 1:
                return False, f"Weight for {indicator} must be between 0 and 1 (current: {weight})"
        
        return True, ""
    
    def validate_categories(self, categories: List[Dict]) -> Tuple[bool, List[str]]:
        """Validate category configuration."""
        errors = []
        
        for i, category in enumerate(categories):
            if not category.get('name', '').strip():
                errors.append(f"Category {i+1}: Name is required")
            
            min_growth = category.get('growth_constraint_min')
            max_growth = category.get('growth_constraint_max')
            
            if min_growth is not None and max_growth is not None:
                if min_growth >= max_growth:
                    errors.append(f"Category {i+1}: Min growth must be less than max growth")
            
            if min_growth is not None and (min_growth < -50 or min_growth > 50):
                errors.append(f"Category {i+1}: Min growth ({min_growth}%) seems unrealistic")
            
            if max_growth is not None and (max_growth < -50 or max_growth > 50):
                errors.append(f"Category {i+1}: Max growth ({max_growth}%) seems unrealistic")
        
        return len(errors) == 0, errors
    
    def validate_topic_configuration(self, topics: List[Dict]) -> Tuple[bool, List[str]]:
        """Validate topic configuration."""
        errors = []
        
        if not topics:
            errors.append("No topics configured")
            return False, errors
        
        active_topics = [t for t in topics if t.get('active', True)]
        if not active_topics:
            errors.append("No active topics found")
        
        for i, topic in enumerate(topics):
            if not topic.get('topic', '').strip():
                errors.append(f"Topic {i+1}: Topic text is required")
            
            if topic.get('type') not in ['AI', 'Manual']:
                errors.append(f"Topic {i+1}: Invalid type (must be AI or Manual)")
        
        return len(errors) == 0, errors
    
    def _is_method_available(self, method: str, unified_data: Dict, config: Dict) -> bool:
        """Check if a method can be executed with current data."""
        from ...forecasting.validation import validate_method_data
        
        selected_countries = config.get('selected_countries', []) if method == 'Country-Specific' else None
        validation = validate_method_data(method, unified_data, selected_countries)
        
        return validation['valid']
    
    def _get_available_countries(self, unified_data: Dict) -> List[str]:
        """Get list of countries with market data."""
        country_market = unified_data.get('market_value', {}).get('country', pd.DataFrame())
        if country_market.empty or 'country' not in country_market.columns:
            return []
        
        return list(country_market['country'].unique())
