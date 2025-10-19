"""
Ollama LLM provider implementation for local models.
"""

import os
import json
import logging
import requests
from typing import Dict, Any, Optional
from ..core.base import BaseLLM
from ..core.exceptions import ProviderException

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLM):
    """Ollama provider for local LLM models."""
    
    def __init__(
        self,
        model_name: str = "llama2",
        base_url: Optional[str] = None,
        timeout: int = 120
    ):
        """
        Initialize Ollama provider.
        
        Args:
            model_name: Ollama model name
            base_url: Ollama server URL
            timeout: Request timeout in seconds
        """
        super().__init__("Ollama", model_name)
        
        self.base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.timeout = timeout
        
        # Ensure base URL doesn't end with slash
        self.base_url = self.base_url.rstrip('/')
    
    def call(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call Ollama API with prompt.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            max_tokens: Maximum tokens (maps to num_predict)
            temperature: Temperature for generation
            **kwargs: Additional Ollama parameters
            
        Returns:
            Response dictionary
        """
        url = f"{self.base_url}/api/generate"
        
        # Combine prompts
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Prepare request data
        data = {
            "model": self.model_name,
            "prompt": full_prompt,
            "temperature": temperature,
            "stream": False
        }
        
        # Map max_tokens to Ollama's num_predict
        if max_tokens:
            data["options"] = {"num_predict": max_tokens}
        
        # Add any additional parameters
        data.update(kwargs)
        
        try:
            # Make request
            response = requests.post(
                url,
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            # Track usage
            self._log_call()
            
            return {
                'content': result.get('response', ''),
                'model': result.get('model', self.model_name),
                'done': result.get('done', True),
                'context': result.get('context', []),
                'total_duration': result.get('total_duration', 0),
                'load_duration': result.get('load_duration', 0),
                'eval_duration': result.get('eval_duration', 0),
                'usage': {
                    'prompt_tokens': result.get('prompt_eval_count', 0),
                    'completion_tokens': result.get('eval_count', 0),
                    'total_tokens': result.get('prompt_eval_count', 0) + result.get('eval_count', 0)
                }
            }
            
        except requests.exceptions.ConnectionError:
            raise ProviderException(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Please ensure Ollama is running.",
                "Ollama",
                error_code="connection_error"
            )
        except requests.exceptions.Timeout:
            raise ProviderException(
                f"Ollama request timed out after {self.timeout} seconds",
                "Ollama",
                error_code="timeout"
            )
        except Exception as e:
            raise ProviderException(
                f"Ollama API call failed: {str(e)}",
                "Ollama"
            )
    
    def test_connection(self) -> bool:
        """
        Test Ollama connection and model availability.
        
        Returns:
            True if connection successful and model available
        """
        try:
            # Check if Ollama is running
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()
            
            # Check if model is available
            models = response.json().get('models', [])
            model_names = [m.get('name', '').split(':')[0] for m in models]
            
            if self.model_name not in model_names:
                logger.warning(
                    f"Model {self.model_name} not found in Ollama. "
                    f"Available models: {model_names}"
                )
                return False
            
            # Try a simple generation
            result = self.call(
                prompt="Hi",
                system_prompt="Reply with 'Hello' only.",
                max_tokens=10,
                temperature=0
            )
            
            return bool(result.get('content'))
            
        except Exception as e:
            logger.error(f"Ollama connection test failed: {str(e)}")
            return False
    
    def list_models(self) -> list:
        """
        List available Ollama models.
        
        Returns:
            List of available model names
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()
            
            models = response.json().get('models', [])
            return [m.get('name', '') for m in models]
            
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {str(e)}")
            return []