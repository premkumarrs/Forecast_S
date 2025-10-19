"""
Custom exceptions for LLM module.
"""


class LLMException(Exception):
    """Base exception for LLM-related errors."""
    
    def __init__(self, message: str, provider: str = None, details: dict = None):
        """
        Initialize LLM exception.
        
        Args:
            message: Error message
            provider: Provider name where error occurred
            details: Additional error details
        """
        self.provider = provider
        self.details = details or {}
        
        if provider:
            message = f"[{provider}] {message}"
        
        super().__init__(message)


class ProviderException(LLMException):
    """Exception raised for provider-specific errors."""
    
    def __init__(self, message: str, provider: str, error_code: str = None, **kwargs):
        """
        Initialize provider exception.
        
        Args:
            message: Error message
            provider: Provider name
            error_code: Provider-specific error code
            **kwargs: Additional arguments for LLMException
        """
        if error_code:
            kwargs.setdefault('details', {})['error_code'] = error_code
        
        super().__init__(message, provider, **kwargs)


class ResponseParseException(LLMException):
    """Exception raised when response parsing fails."""
    
    def __init__(self, message: str, raw_response: str = None, **kwargs):
        """
        Initialize response parse exception.
        
        Args:
            message: Error message
            raw_response: The raw response that failed to parse
            **kwargs: Additional arguments for LLMException
        """
        if raw_response:
            kwargs.setdefault('details', {})['raw_response'] = raw_response[:500]  # Truncate
        
        super().__init__(message, **kwargs)


class RateLimitException(ProviderException):
    """Exception raised when rate limit is hit."""
    
    def __init__(self, provider: str, retry_after: int = None, **kwargs):
        """
        Initialize rate limit exception.
        
        Args:
            provider: Provider name
            retry_after: Seconds to wait before retry
            **kwargs: Additional arguments
        """
        message = f"Rate limit exceeded for {provider}"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
            kwargs.setdefault('details', {})['retry_after'] = retry_after
        
        super().__init__(message, provider, error_code='rate_limit', **kwargs)