"""
Base prompt templates and builder.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BasePromptBuilder(ABC):
    """Abstract base class for prompt builders."""
    
    def __init__(self, name: str):
        """
        Initialize prompt builder.
        
        Args:
            name: Builder name
        """
        self.name = name
    
    @abstractmethod
    def build(self, **kwargs) -> str:
        """
        Build the prompt.
        
        Args:
            **kwargs: Prompt-specific parameters
            
        Returns:
            Built prompt string
        """
        pass
    
    def validate_params(self, required: list, provided: dict) -> bool:
        """
        Validate that required parameters are provided.
        
        Args:
            required: List of required parameter names
            provided: Dictionary of provided parameters
            
        Returns:
            True if all required parameters are present
        """
        for param in required:
            if param not in provided:
                raise ValueError(f"Missing required parameter: {param}")
        return True
    
    def format_json_requirements(self) -> str:
        """
        Standard JSON format requirements text.
        
        Returns:
            JSON requirements string
        """
        return """CRITICAL JSON RESPONSE REQUIREMENTS:
- You MUST respond with valid JSON only
- Do not include any text before or after the JSON
- No markdown code blocks, no explanations, no additional text
- Use double quotes for all strings
- No trailing commas
- No comments in JSON
- Ensure the JSON is valid and parseable"""
    
    def format_temporal_guidelines(self) -> str:
        """
        Standard temporal impact guidelines.
        
        Returns:
            Temporal guidelines string
        """
        return """Market Impact Guidelines:
- HIGH_RELEVANCE: Strategic investments, infrastructure changes, regulatory shifts, technology breakthroughs, business model changes, major market announcements
- MEDIUM_RELEVANCE: Supply disruptions, quarterly results, executive changes, product launches, pricing actions, partnerships
- LOW_RELEVANCE: General commentary, industry analysis without specific market impact
- NO_EFFECT: Opinions, rumors, duplicate news, unrelated content"""