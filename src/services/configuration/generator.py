"""
Configuration generation module for creating topics and categories.
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class ConfigurationGenerator:
    """Handles generation of configuration elements like topics and categories."""
    
    def __init__(self):
        self.llm_analyst = None
    
    def generate_topics(self, market_name: str, num_topics: int = 50) -> List[str]:
        """Generate topics using LLM."""
        try:
            if not self.llm_analyst:
                from ...llm.analyst import SimpleLLMAnalyst as LLM_Analyst
                self.llm_analyst = LLM_Analyst(market_name)
            
            topics = self.llm_analyst.generate_topics(num_topics)
            logger.info(f"LLM generated {len(topics)} topics for {market_name}")
            
            # Ensure we return strings
            valid_topics = [str(topic).strip() for topic in topics if topic and str(topic).strip()]
            
            if valid_topics:
                return valid_topics
            else:
                logger.warning("LLM returned empty topics list, using fallback")
                raise ValueError("Empty topics from LLM")
            
        except Exception as e:
            logger.warning(f"LLM topic generation failed: {e}, using fallback topics")
            return self._get_fallback_topics(market_name, num_topics)
    
    def generate_categories(self, market_name: str, num_categories: int = 8,
                          min_growth: float = -10.0, max_growth: float = 15.0) -> List[Dict]:
        """Generate categories for a market using LLM."""
        try:
            if not self.llm_analyst:
                from ...llm.analyst import SimpleLLMAnalyst as LLM_Analyst
                self.llm_analyst = LLM_Analyst(market_name)
            
            # Generate categories using LLM
            categories_data = self.llm_analyst.generate_categories(num_categories, min_growth, max_growth)
            
            # Ensure proper format
            formatted_categories = []
            for i, category in enumerate(categories_data):
                if isinstance(category, str):
                    # Simple string category
                    formatted_categories.append({
                        'name': category,
                        'description': '',
                        'growth_constraint_min': min_growth,
                        'growth_constraint_max': max_growth,
                        'importance': 'medium'
                    })
                elif isinstance(category, dict):
                    # Structured category from LLM
                    formatted_categories.append({
                        'name': category.get('name', f'Category {i+1}'),
                        'description': category.get('description', ''),
                        'growth_constraint_min': category.get('growth_constraint_min', min_growth),
                        'growth_constraint_max': category.get('growth_constraint_max', max_growth),
                        'importance': category.get('importance', 'medium')
                    })
            
            return formatted_categories
            
        except Exception as e:
            logger.warning(f"LLM category generation failed: {e}, using fallback")
            return self._get_fallback_categories(market_name, min_growth, max_growth)
    
    def create_default_indicator_weights(self, available_indicators: List[str]) -> Dict[str, float]:
        """Create default equal weights for indicators."""
        if not available_indicators:
            return {}
        
        weight_per_indicator = 1.0 / len(available_indicators)
        return {indicator: weight_per_indicator for indicator in available_indicators}
    
    def create_default_topics(self, market_name: str) -> List[Dict]:
        """Create default topics for a market."""
        default_topics = [
            f"{market_name} market trends",
            f"{market_name} industry news", 
            f"{market_name} technology developments",
            f"{market_name} regulatory changes",
            f"{market_name} investment news"
        ]
        
        return [
            {'topic': topic, 'type': 'Manual', 'active': True}
            for topic in default_topics
        ]
    
    def _get_fallback_topics(self, market_name: str, num_topics: int) -> List[str]:
        """Get fallback topics if LLM fails."""
        base_topics = [
            f"{market_name} market trends", f"{market_name} industry news", 
            f"{market_name} technology developments", f"{market_name} regulatory changes",
            f"{market_name} investment news", f"{market_name} partnerships",
            f"{market_name} mergers acquisitions", f"{market_name} innovation",
            f"{market_name} competition", f"{market_name} pricing strategies",
            f"{market_name} customer adoption", f"{market_name} market share",
            f"{market_name} growth opportunities", f"{market_name} challenges",
            f"{market_name} security", f"{market_name} compliance"
        ]
        
        # Pad to requested number of topics
        while len(base_topics) < num_topics:
            base_topics.extend([f"{market_name} analysis", f"{market_name} forecast"])
        
        return base_topics[:num_topics]
    
    def _get_fallback_categories(self, market_name: str, min_growth: float, max_growth: float) -> List[Dict]:
        """Get fallback categories if LLM fails."""
        return [
            {
                'name': f'{market_name} Core Products',
                'description': 'Main product offerings and services',
                'growth_constraint_min': -5.0,
                'growth_constraint_max': 10.0,
                'importance': 'high'
            },
            {
                'name': f'{market_name} Services',
                'description': 'Service offerings and support',
                'growth_constraint_min': -8.0,
                'growth_constraint_max': 12.0,
                'importance': 'medium'
            },
            {
                'name': f'{market_name} Infrastructure',
                'description': 'Infrastructure and platform development',
                'growth_constraint_min': -10.0,
                'growth_constraint_max': 8.0,
                'importance': 'medium'
            },
            {
                'name': f'{market_name} Innovation',
                'description': 'New technologies and innovations',
                'growth_constraint_min': -15.0,
                'growth_constraint_max': 20.0,
                'importance': 'high'
            }
        ]