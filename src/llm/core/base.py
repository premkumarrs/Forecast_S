"""
Base class for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, name: str, model_name: str):
        """
        Initialize base LLM.
        
        Args:
            name: Provider name
            model_name: Model identifier
        """
        self.name = name
        self.model_name = model_name
        self.total_tokens_used = 0
        self.total_calls = 0
    
    @abstractmethod
    def call(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call the LLM with a prompt.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            max_tokens: Maximum tokens in response
            temperature: Temperature for generation
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Response dictionary with 'content' and metadata
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test if the LLM connection is working.
        
        Returns:
            True if connection is successful
        """
        pass
    
    def parse_response(self, raw_response: str) -> Any:
        """
        Parse LLM response to extract structured data.
        
        Args:
            raw_response: Raw response string
            
        Returns:
            Parsed response dictionary
        """
        # Default implementation - can be overridden
        return self._default_parse(raw_response)
    
    def _default_parse(self, raw_response: str) -> Any:
        """
        Default response parsing logic.
        
        Args:
            raw_response: Raw response string
            
        Returns:
            Parsed response (dict, list, or wrapped in dict)
        """
        import json
        import re
        
        # First try to parse the entire response as JSON
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON array from the response
        array_pattern = r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'
        array_matches = re.findall(array_pattern, raw_response)
        
        for match in array_matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Try to extract JSON object from the response
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, raw_response)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # If no valid JSON found, return as text
        return {'text': raw_response}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Dictionary with usage stats
        """
        return {
            'provider': self.name,
            'model': self.model_name,
            'total_calls': self.total_calls,
            'total_tokens': self.total_tokens_used
        }
    
    def reset_stats(self):
        """Reset usage statistics."""
        self.total_tokens_used = 0
        self.total_calls = 0
    
    def _log_call(self, tokens_used: int = 0):
        """
        Log an API call.
        
        Args:
            tokens_used: Number of tokens used
        """
        self.total_calls += 1
        self.total_tokens_used += tokens_used