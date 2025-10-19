"""
Simplified LLM analyst using the simple LLM interface instead of complex provider pattern.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

from .providers.provider_factory import call_llm, test_llm_connection
from .categories import load_market_categories
from .prompts import build_system_prompt, build_country_system_prompt

load_dotenv('config/.env')
logger = logging.getLogger(__name__)


class SimpleLLMAnalyst:
    """
    Simplified LLM-powered analyst for news analysis.
    Uses direct function calls instead of complex provider pattern.
    """
    
    def __init__(self, market_name: str, categories: Optional[List[Dict[str, Any]]] = None):
        """Initialize the simplified analyst."""
        self.market_name = market_name
        
        # Load categories
        if categories is not None:
            self.categories = categories
        else:
            self.categories = load_market_categories(self.market_name)
        
        # Build system prompt
        self.system_prompt = build_system_prompt(self.market_name, self.categories)
        
        # Concurrency settings
        self.max_concurrent_requests = int(os.getenv('LLM_MAX_CONCURRENT', '10'))
        self.request_delay = float(os.getenv('LLM_REQUEST_DELAY', '0.05'))

    def analyze_headlines(self, headlines: List[Dict[str, Any]], forecast_years: int = None, country: Optional[str] = None, progress_callback=None) -> pd.DataFrame:
        """
        Analyze headlines using LLM with configurable concurrency.
        
        Args:
            headlines: List of headline dictionaries with 'title' key
            forecast_years: Number of years in forecast horizon
            country: Optional country name for localized analysis
            progress_callback: Function to call with progress updates
            
        Returns:
            pd.DataFrame: Analysis results
        """
        if not headlines:
            return pd.DataFrame()
        
        # Choose appropriate prompt based on country context
        if country:
            logger.info(f"Using COUNTRY-SPECIFIC prompt for: {country}")
            system_prompt = build_country_system_prompt(self.market_name, country, self.categories, forecast_years)
        else:
            logger.info("Using GLOBAL prompt (no country context)")
            system_prompt = build_system_prompt(self.market_name, self.categories, forecast_years)
        
        # Use concurrent processing
        results = []
        with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
            # Submit all headlines for parallel processing
            future_to_headline = {
                executor.submit(self._analyze_single_headline, headline, system_prompt): headline
                for headline in headlines
            }
            
            # Collect results as they complete
            completed = 0
            total = len(headlines)
            for future in as_completed(future_to_headline):
                try:
                    analysis = future.result()
                    results.append(analysis)
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total, "Analyzing headlines")
                except Exception as e:
                    logger.error(f"Failed to analyze headline: {e}")
                    completed += 1
        
        return pd.DataFrame(results)

    def _analyze_single_headline(self, headline: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
        """Analyze a single headline."""
        title = headline.get('title', '')
        if not title:
            return self._get_empty_analysis()
        
        # Add delay for rate limiting
        if self.request_delay > 0:
            time.sleep(self.request_delay)
        
        try:
            # Make LLM request
            llm_response = call_llm(title, system_prompt, max_tokens=150)
            
            # Extract the actual response (it's nested under 'response' key)
            parsed_response = llm_response.get('response', {})
            
            # Log the parsed response for debugging
            if parsed_response and isinstance(parsed_response, dict):
                logger.debug(f"LLM parsed response: category={parsed_response.get('category')}, "
                           f"growth_rate={parsed_response.get('growth_rate')}")
            
            # Handle error cases
            if 'error' in llm_response:
                logger.error(f"LLM error for headline: {llm_response['error']}")
                return self._get_empty_analysis(title, headline.get('date', ''))
            
            # Convert response to analysis format
            analysis = {
                'title': title,
                'date': headline.get('date', ''),
                'category': parsed_response.get('category', 'Neutral/Noise'),
                'growth_rate': float(parsed_response.get('growth_rate', 0.0)),
                'reason': parsed_response.get('reason', 'Analysis completed'),
                'relevant': 1 if parsed_response.get('growth_rate', 0.0) != 0.0 else 0
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze headline: {str(e)[:100]}")
            return self._get_empty_analysis(title, headline.get('date', ''))

    def _get_empty_analysis(self, title: str = '', date: str = '') -> Dict[str, Any]:
        """Return empty analysis for failed cases."""
        return {
            'title': title,
            'date': date,
            'category': 'Neutral/Noise',
            'growth_rate': 0.0,
            'reason': 'Analysis failed - using defaults',
            'relevant': 0
        }

    def analyze_headlines_concurrent(self, headlines: List[Dict[str, Any]], forecast_years: int = None) -> pd.DataFrame:
        """
        Concurrent headline analysis (simplified to sequential for now).
        Can be enhanced with ThreadPoolExecutor if needed.
        """
        return self.analyze_headlines(headlines, forecast_years)

    def generate_categories(self, num_categories: int, min_growth_rate: float, max_growth_rate: float) -> List[Dict[str, Any]]:
        """Generate categories using LLM."""
        from .prompts import build_category_generation_prompt
        
        prompt = build_category_generation_prompt(self.market_name, num_categories, min_growth_rate, max_growth_rate)
        
        try:
            response = call_llm("Generate categories", prompt, max_tokens=4096)
            # The response is nested under 'response' key
            parsed_response = response.get('response', {})
            
            # Handle different response formats
            if isinstance(parsed_response, list):
                categories = parsed_response
            elif isinstance(parsed_response, dict):
                categories = parsed_response.get('categories', [])
            else:
                categories = []
            
            # Ensure each category has required fields
            validated_categories = []
            for cat in categories:
                if isinstance(cat, dict) and 'name' in cat:
                    validated_categories.append({
                        'name': cat['name'],
                        'description': cat.get('description', ''),
                        'growth_constraint_min': cat.get('growth_constraint_min', min_growth_rate),
                        'growth_constraint_max': cat.get('growth_constraint_max', max_growth_rate)
                    })
            
            return validated_categories
            
        except Exception as e:
            logger.error(f"Failed to generate categories: {e}")
            return []

    def generate_topics(self, num_topics: int = 50) -> List[str]:
        """Generate topics using LLM."""
        from .prompts import build_topic_generation_prompt
        
        prompt = build_topic_generation_prompt(self.market_name, num_topics)
        
        try:
            response = call_llm("Generate topics", prompt, max_tokens=2048)
            logger.info(f"LLM response for topic generation: {response}")
            
            # The response is nested under 'response' key
            parsed_response = response.get('response', {})
            
            # Handle different response formats
            if isinstance(parsed_response, list):
                topics = parsed_response
            elif isinstance(parsed_response, dict):
                topics = parsed_response.get('topics', [])
            else:
                topics = []
            logger.info(f"Extracted topics: {topics}")
            
            # Ensure topics are strings and not empty
            valid_topics = [str(topic).strip() for topic in topics if topic and str(topic).strip()]
            logger.info(f"Valid topics after filtering: {valid_topics}")
            
            return valid_topics
            
        except Exception as e:
            logger.error(f"Failed to generate topics: {e}")
            # Return fallback topics instead of empty list
            fallback_topics = [
                f"{self.market_name} trends",
                f"{self.market_name} analysis", 
                f"{self.market_name} innovation",
                f"{self.market_name} investment",
                f"{self.market_name} technology",
                f"{self.market_name} market",
                f"{self.market_name} growth",
                f"{self.market_name} industry"
            ]
            return fallback_topics[:num_topics]

    def test_connection(self) -> bool:
        """Test LLM connection."""
        return test_llm_connection()


# Backward compatibility alias
LLM_Analyst = SimpleLLMAnalyst