"""
Factory for creating LLM provider instances.
"""

import os
import logging
from typing import Optional, Dict, Type, Any
from ..core.base import BaseLLM
from ..core.exceptions import ProviderException
from .openai_provider import OpenAIProvider
from .ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)


class ProviderFactory:
    """Factory class for creating LLM provider instances."""
    
    # Registry of available providers
    _providers: Dict[str, Type[BaseLLM]] = {
        'openai': OpenAIProvider,
        'ollama': OllamaProvider,
    }
    
    @classmethod
    def create_provider(
        cls,
        provider_name: Optional[str] = None,
        **kwargs
    ) -> BaseLLM:
        """
        Create an LLM provider instance.
        
        Args:
            provider_name: Name of the provider (or from environment)
            **kwargs: Provider-specific configuration
            
        Returns:
            LLM provider instance
            
        Raises:
            ProviderException: If provider not found or configuration invalid
        """
        # Get provider name from environment if not specified
        if not provider_name:
            provider_name = os.getenv('LLM_PROVIDER', 'ollama').lower()
        else:
            provider_name = provider_name.lower()
        
        # Check if provider exists
        if provider_name not in cls._providers:
            available = list(cls._providers.keys())
            raise ProviderException(
                f"Unknown provider: {provider_name}. Available: {available}",
                provider_name
            )
        
        # Get provider class
        provider_class = cls._providers[provider_name]
        
        # Get provider-specific configuration from environment
        config = cls._get_provider_config(provider_name)
        config.update(kwargs)  # Override with explicit kwargs
        
        try:
            # Create provider instance
            return provider_class(**config)
        except Exception as e:
            raise ProviderException(
                f"Failed to create {provider_name} provider: {str(e)}",
                provider_name
            )
    
    @classmethod
    def _get_provider_config(cls, provider_name: str) -> Dict:
        """
        Get provider-specific configuration from environment.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Configuration dictionary
        """
        config = {}
        
        if provider_name == 'openai':
            config['api_key'] = os.getenv('OPENAI_API_KEY')
            config['model_name'] = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
            config['base_url'] = os.getenv('OPENAI_BASE_URL')
            # Optional Azure OpenAI version
            api_version = os.getenv('OPENAI_API_VERSION')
            if api_version:
                config['api_version'] = api_version
            
        elif provider_name == 'ollama':
            config['model_name'] = os.getenv('OLLAMA_MODEL', 'llama2')
            config['base_url'] = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            
        # Add more providers here as needed
        
        return {k: v for k, v in config.items() if v is not None}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseLLM]):
        """
        Register a new provider type.
        
        Args:
            name: Provider name
            provider_class: Provider class (must inherit from BaseLLM)
        """
        if not issubclass(provider_class, BaseLLM):
            raise TypeError(f"{provider_class} must inherit from BaseLLM")
        
        cls._providers[name.lower()] = provider_class
        logger.info(f"Registered new LLM provider: {name}")
    
    @classmethod
    def get_available_providers(cls) -> list:
        """
        Get list of available provider names.
        
        Returns:
            List of provider names
        """
        return list(cls._providers.keys())
    
    @classmethod
    def test_all_providers(cls) -> Dict[str, bool]:
        """
        Test connection for all configured providers.
        
        Returns:
            Dictionary mapping provider names to connection status
        """
        results = {}
        
        for provider_name in cls._providers:
            try:
                provider = cls.create_provider(provider_name)
                results[provider_name] = provider.test_connection()
            except Exception as e:
                logger.error(f"Failed to test {provider_name}: {str(e)}")
                results[provider_name] = False
        
        return results


# Backward compatibility function
def call_llm(
    prompt: str,
    system_prompt: str,
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    Backward compatibility wrapper for LLM calls.
    
    Args:
        prompt: User prompt
        system_prompt: System prompt
        max_tokens: Maximum tokens
        
    Returns:
        Response dictionary
    """
    try:
        # Create provider based on environment
        provider = ProviderFactory.create_provider()
        
        # Make call
        response = provider.call(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens
        )
        
        # Parse response
        parsed = provider.parse_response(response.get('content', ''))
        
        return {
            'response': parsed,
            'raw_response': response.get('content', ''),
            'model': response.get('model'),
            'usage': response.get('usage', {})
        }
        
    except Exception as e:
        logger.error(f"LLM call failed: {str(e)}")
        # Return fallback response
        return {
            'response': {},
            'raw_response': '',
            'error': str(e)
        }


def test_llm_connection() -> bool:
    """
    Backward compatibility wrapper for testing LLM connection.
    
    Returns:
        True if any provider is working
    """
    try:
        provider = ProviderFactory.create_provider()
        return provider.test_connection()
    except:
        return False
