"""
Base class for forecast adjustments.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Tuple, Optional


class BaseAdjustment(ABC):
    """Abstract base class for forecast adjustments."""
    
    def __init__(self, weight: float = 1.0):
        """
        Initialize base adjustment.
        
        Args:
            weight: Weight of this adjustment type (0-1)
        """
        self.weight = weight
        self.adjustment_value = 0.0
        self.details = {}
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, config: Dict) -> float:
        """
        Calculate the adjustment value.
        
        Args:
            data: Input data for adjustment calculation
            config: Configuration parameters
            
        Returns:
            Adjustment value (as percentage, e.g., 0.1 for 10%)
        """
        pass
    
    @abstractmethod
    def apply(
        self,
        baseline_df: pd.DataFrame,
        data: pd.DataFrame,
        config: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply adjustment to baseline forecast.
        
        Args:
            baseline_df: Baseline forecast DataFrame
            data: Input data for adjustment
            config: Configuration parameters
            
        Returns:
            Tuple of (adjusted DataFrame, adjustment details)
        """
        pass
    
    def validate_adjustment(self, value: float, min_val: float, max_val: float) -> float:
        """
        Validate and cap adjustment value within bounds.
        
        Args:
            value: Raw adjustment value
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Validated adjustment value
        """
        return max(min_val, min(max_val, value))
    
    def get_weighted_adjustment(self, raw_adjustment: float) -> float:
        """
        Apply weight to raw adjustment.
        
        Args:
            raw_adjustment: Raw adjustment value
            
        Returns:
            Weighted adjustment value
        """
        return raw_adjustment * self.weight
    
    def get_details(self) -> Dict:
        """
        Get adjustment details for transparency.
        
        Returns:
            Dictionary with adjustment details
        """
        return {
            'weight': self.weight,
            'adjustment_value': self.adjustment_value,
            **self.details
        }