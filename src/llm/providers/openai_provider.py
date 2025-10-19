"""
OpenAI LLM provider implementation.
"""

import os
import logging
from typing import Dict, Any, Optional
from ..core.base import BaseLLM
from ..core.exceptions import ProviderException, RateLimitException

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLM):
    """OpenAI API provider for LLM calls."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        base_url: Optional[str] = None,
        api_version: Optional[str] = None
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (or from environment)
            model_name: Model to use
            base_url: Optional base URL for API
        """
        super().__init__("OpenAI", model_name)
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = (base_url or os.getenv('OPENAI_BASE_URL') or '').rstrip('/') if (base_url or os.getenv('OPENAI_BASE_URL')) else None
        self.api_version = api_version or os.getenv('OPENAI_API_VERSION')
        
        if not self.api_key:
            raise ProviderException(
                "OpenAI API key not provided",
                "OpenAI",
                error_code="missing_api_key"
            )
        
        self._client = None
    
    def _is_azure_mode(self) -> bool:
        """Detect if we should use AzureOpenAI client."""
        if self.base_url and 'openai.azure.com' in self.base_url:
            return True
        if os.getenv('AZURE_OPENAI_ENDPOINT'):
            return True
        if os.getenv('OPENAI_API_VERSION'):
            # Azure SDK requires api_version; treat presence as azure signal
            return True
        return False

    def _derive_azure_endpoint(self) -> Optional[str]:
        """Derive Azure endpoint from base_url if possible.

        If base_url looks like https://<resource>.openai.azure.com/openai/deployments/<deployment>
        we return https://<resource>.openai.azure.com/
        """
        endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        if endpoint:
            return endpoint.rstrip('/') + '/'
        if not self.base_url:
            return None
        marker = '/openai/deployments'
        if marker in self.base_url:
            return self.base_url.split(marker)[0].rstrip('/') + '/'
        # Fallback: assume base_url is already the endpoint root
        return self.base_url.rstrip('/') + '/'

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                if self._is_azure_mode():
                    from openai import AzureOpenAI
                    azure_endpoint = self._derive_azure_endpoint()
                    version = self.api_version or '2025-01-01-preview'
                    self._client = AzureOpenAI(
                        api_key=self.api_key,
                        azure_endpoint=azure_endpoint,
                        api_version=version
                    )
                else:
                    from openai import OpenAI
                    self._client = OpenAI(
                        api_key=self.api_key,
                        base_url=self.base_url
                    )
            except ImportError:
                raise ProviderException(
                    "OpenAI package not installed. Run: pip install openai",
                    "OpenAI",
                    error_code="missing_package"
                )
        return self._client
    
    def call(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call OpenAI API with prompt.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            max_tokens: Maximum tokens in response
            temperature: Temperature for generation
            **kwargs: Additional OpenAI parameters
            
        Returns:
            Response dictionary
        """
        client = self._get_client()
        
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Prepare parameters
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature
            }
            
            if max_tokens:
                params["max_tokens"] = max_tokens
            
            # Add any additional parameters
            params.update(kwargs)
            
            # Make API call
            response = client.chat.completions.create(**params)
            
            # Extract content
            content = response.choices[0].message.content
            
            # Track usage
            if hasattr(response, 'usage'):
                tokens_used = response.usage.total_tokens
                self._log_call(tokens_used)
            else:
                self._log_call()
            
            return {
                'content': content,
                'model': response.model,
                'finish_reason': response.choices[0].finish_reason,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens if hasattr(response, 'usage') else 0,
                    'completion_tokens': response.usage.completion_tokens if hasattr(response, 'usage') else 0,
                    'total_tokens': response.usage.total_tokens if hasattr(response, 'usage') else 0
                }
            }
            
        except Exception as e:
            if "rate_limit" in str(e).lower():
                raise RateLimitException("OpenAI")
            else:
                raise ProviderException(
                    f"OpenAI API call failed: {str(e)}",
                    "OpenAI"
                )
    
    def test_connection(self) -> bool:
        """
        Test OpenAI connection.
        
        Returns:
            True if connection successful
        """
        try:
            response = self.call(
                prompt="Hello",
                system_prompt="You are a helpful assistant. Reply with 'Hi' only.",
                max_tokens=10,
                temperature=0
            )
            return bool(response.get('content'))
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {str(e)}")
            return False
