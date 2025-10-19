"""
Base service class for all service layer components.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseService(ABC):
    """Abstract base class for all services."""
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        """
        Initialize base service.
        
        Args:
            name: Service name for logging
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"services.{name}")
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the service if not already initialized."""
        if not self._initialized:
            self.logger.info(f"Initializing {self.name} service")
            self._initialize()
            self._initialized = True
    
    @abstractmethod
    def _initialize(self) -> None:
        """Service-specific initialization logic."""
        pass
    
    def validate_input(self, data: Any, validation_rules: Optional[Dict] = None) -> bool:
        """
        Validate input data against rules.
        
        Args:
            data: Input data to validate
            validation_rules: Optional validation rules
            
        Returns:
            True if valid, raises exception otherwise
        """
        if validation_rules:
            return self._apply_validation_rules(data, validation_rules)
        return True
    
    def _apply_validation_rules(self, data: Any, rules: Dict) -> bool:
        """
        Apply validation rules to data.
        
        Args:
            data: Data to validate
            rules: Validation rules dictionary
            
        Returns:
            True if all rules pass
        """
        for field, rule in rules.items():
            if 'required' in rule and rule['required']:
                if not hasattr(data, field) or getattr(data, field) is None:
                    raise ValueError(f"Required field missing: {field}")
            
            if 'type' in rule:
                expected_type = rule['type']
                if hasattr(data, field):
                    value = getattr(data, field)
                    if value is not None and not isinstance(value, expected_type):
                        raise TypeError(
                            f"Field {field} must be of type {expected_type.__name__}"
                        )
        
        return True
    
    def handle_error(self, error: Exception, context: Optional[str] = None) -> None:
        """
        Handle and log errors consistently.
        
        Args:
            error: The exception that occurred
            context: Optional context information
        """
        error_msg = f"Error in {self.name}"
        if context:
            error_msg += f" during {context}"
        error_msg += f": {str(error)}"
        
        self.logger.error(error_msg, exc_info=True)
        
    def get_status(self) -> Dict:
        """
        Get current service status.
        
        Returns:
            Dictionary with status information
        """
        return {
            'name': self.name,
            'initialized': self._initialized,
            'config': self.config
        }
    
    def cleanup(self) -> None:
        """Cleanup service resources."""
        self.logger.info(f"Cleaning up {self.name} service")
        self._cleanup()
        self._initialized = False
    
    def _cleanup(self) -> None:
        """Service-specific cleanup logic."""
        pass