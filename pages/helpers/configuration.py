"""
Helper functions for Configuration page.
"""

import streamlit as st
from typing import List, Tuple
from src.services.configuration import ConfigurationService


def generate_topics_with_llm(market_name: str, num_topics: int = 7) -> List[str]:
    """
    Generate topics using LLM.
    
    Args:
        market_name: Market name
        num_topics: Number of topics to generate
        
    Returns:
        List of generated topics
    """
    service = ConfigurationService()
    return service.generate_topics_with_llm(market_name, num_topics)


def get_available_countries_from_session() -> List[str]:
    """
    Get available countries from session state.
    
    Returns:
        List of country names
    """
    unified_data = st.session_state.get('unified_data', {})
    market_data = unified_data.get('market_value', {})
    country_data = market_data.get('country', None)
    
    if country_data is not None and not country_data.empty and 'country' in country_data.columns:
        return sorted(list(country_data['country'].unique()))
    
    return []


def validate_configuration() -> Tuple[bool, List[str]]:
    """
    Validate current configuration.
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    config = st.session_state.get('config', {})
    unified_data = st.session_state.get('unified_data', {})
    
    # Validate with service
    service = ConfigurationService()
    is_valid, service_errors = service.validate_configuration(config, unified_data)
    errors.extend(service_errors)
    
    # Additional validation
    if not config.get('market_name'):
        errors.append("Market name is required")
    
    if not config.get('forecast_method'):
        errors.append("Forecast method must be selected")
    
    return len(errors) == 0, errors