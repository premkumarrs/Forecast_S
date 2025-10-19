"""
Unified prompt builder for all prompt types.
"""

from typing import Dict, Any, Optional
from .base_prompts import BasePromptBuilder
from .analysis_prompts import build_system_prompt, build_country_system_prompt
from .category_prompts import (
    build_category_generation_prompt,
    build_topic_generation_prompt
)


class PromptBuilder:
    """Unified builder for all prompt types."""
    
    def __init__(self):
        """Initialize prompt builder."""
        self.builders = {}
    
    def build_analysis_prompt(
        self,
        market_name: str,
        categories: list,
        forecast_years: Optional[int] = None,
        country: Optional[str] = None
    ) -> str:
        """
        Build analysis prompt (global or country-specific).
        
        Args:
            market_name: Market name
            categories: List of categories
            forecast_years: Forecast horizon
            country: Optional country for country-specific prompt
            
        Returns:
            Built prompt
        """
        if country:
            return build_country_system_prompt(
                market_name,
                country,
                categories,
                forecast_years
            )
        else:
            return build_system_prompt(
                market_name,
                categories,
                forecast_years
            )
    
    def build_category_prompt(
        self,
        market_name: str,
        num_categories: int = 10,
        min_growth: float = -10.0,
        max_growth: float = 50.0
    ) -> str:
        """
        Build category generation prompt.
        
        Args:
            market_name: Market name
            num_categories: Number of categories to generate
            min_growth: Minimum growth rate
            max_growth: Maximum growth rate
            
        Returns:
            Built prompt
        """
        return build_category_generation_prompt(
            market_name,
            num_categories,
            min_growth,
            max_growth
        )
    
    def build_topic_prompt(
        self,
        market_name: str,
        num_topics: int = 20
    ) -> str:
        """
        Build topic generation prompt.
        
        Args:
            market_name: Market name
            num_topics: Number of topics to generate
            
        Returns:
            Built prompt
        """
        return build_topic_generation_prompt(
            market_name,
            num_topics
        )
    
    def build(
        self,
        prompt_type: str,
        **kwargs
    ) -> str:
        """
        Build any type of prompt.
        
        Args:
            prompt_type: Type of prompt ('analysis', 'category', 'topic')
            **kwargs: Prompt-specific parameters
            
        Returns:
            Built prompt
            
        Raises:
            ValueError: If prompt_type is unknown
        """
        builders = {
            'analysis': self.build_analysis_prompt,
            'category': self.build_category_prompt,
            'topic': self.build_topic_prompt
        }
        
        builder = builders.get(prompt_type)
        if not builder:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        return builder(**kwargs)