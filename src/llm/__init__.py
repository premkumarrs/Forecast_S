"""
LLM integration module for market analysis - simplified interface.
"""

# Import from new modular structure for backward compatibility
from .providers.provider_factory import call_llm, test_llm_connection
from .analyst import SimpleLLMAnalyst, LLM_Analyst

# Also expose new modular components
from .core import BaseLLM, LLMException
from .providers import OpenAIProvider, OllamaProvider, ProviderFactory

__all__ = [
    # Backward compatibility exports
    'call_llm', 
    'test_llm_connection',
    'SimpleLLMAnalyst', 
    'LLM_Analyst',
    
    # New modular components
    'BaseLLM',
    'LLMException',
    'OpenAIProvider',
    'OllamaProvider',
    'ProviderFactory'
]