"""
Custom exceptions for service layer.
"""


class ServiceException(Exception):
    """Base exception for service layer errors."""
    
    def __init__(self, message: str, service_name: str = None, details: dict = None):
        """
        Initialize service exception.
        
        Args:
            message: Error message
            service_name: Name of the service where error occurred
            details: Additional error details
        """
        self.service_name = service_name
        self.details = details or {}
        
        if service_name:
            message = f"[{service_name}] {message}"
        
        super().__init__(message)


class ValidationException(ServiceException):
    """Exception raised for validation failures."""
    
    def __init__(self, message: str, field: str = None, **kwargs):
        """
        Initialize validation exception.
        
        Args:
            message: Error message
            field: Field that failed validation
            **kwargs: Additional arguments for ServiceException
        """
        if field:
            kwargs.setdefault('details', {})['field'] = field
            message = f"Validation failed for field '{field}': {message}"
        
        super().__init__(message, **kwargs)


class DataException(ServiceException):
    """Exception raised for data-related errors."""
    
    def __init__(self, message: str, data_type: str = None, **kwargs):
        """
        Initialize data exception.
        
        Args:
            message: Error message
            data_type: Type of data that caused error
            **kwargs: Additional arguments for ServiceException
        """
        if data_type:
            kwargs.setdefault('details', {})['data_type'] = data_type
            message = f"Data error for type '{data_type}': {message}"
        
        super().__init__(message, **kwargs)


class ConfigurationException(ServiceException):
    """Exception raised for configuration errors."""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        """
        Initialize configuration exception.
        
        Args:
            message: Error message
            config_key: Configuration key that caused error
            **kwargs: Additional arguments for ServiceException
        """
        if config_key:
            kwargs.setdefault('details', {})['config_key'] = config_key
            message = f"Configuration error for key '{config_key}': {message}"
        
        super().__init__(message, **kwargs)