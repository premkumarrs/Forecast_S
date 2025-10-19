"""
Temporal decay calculations for forecast adjustments.
"""

import numpy as np
from typing import Optional


class TemporalDecay:
    """Calculate temporal decay factors for forecast adjustments."""
    
    def __init__(
        self,
        short_term_horizon: int = None,  # Deprecated, kept for compatibility
        long_term_decay_rate: float = 0.6
    ):
        """
        Initialize temporal decay calculator.
        
        Args:
            short_term_horizon: Deprecated parameter, kept for compatibility
            long_term_decay_rate: Decay rate for exponential decay (default 0.6)
        """
        # Keep for backwards compatibility but not used
        self.short_term_horizon = short_term_horizon
        self.long_term_decay_rate = long_term_decay_rate
    
    
    def calculate_exponential_decay(
        self,
        years_from_start: int,
        total_forecast_years: int
    ) -> float:
        """
        Calculate exponential decay factor for news impact.
        
        News effects decay exponentially with a half-life approach over the forecast horizon.
        
        Args:
            years_from_start: Number of years from forecast start
            total_forecast_years: Total years in forecast period
            
        Returns:
            Decay factor (0-1)
        """
        # Exponential decay with half-life
        # At decay_point * total_years, the effect is halved
        decay_point = total_forecast_years * self.long_term_decay_rate
        
        if decay_point <= 0:
            return 1.0
        
        # Half-life formula: 0.5^(t/half_life)
        decay = 0.5 ** (years_from_start / decay_point)
        
        return decay
    
    def calculate_long_term_decay(
        self,
        years_from_start: int,
        total_forecast_years: int
    ) -> float:
        """
        Deprecated: Use calculate_exponential_decay() instead.
        
        This method is kept for backwards compatibility.
        """
        return self.calculate_exponential_decay(years_from_start, total_forecast_years)
    
    def calculate_news_decay(
        self,
        years_from_start: int,
        total_forecast_years: int
    ) -> float:
        """
        Calculate news decay factor using exponential decay.
        
        This is a convenience method that uses exponential decay as the
        single decay profile for news adjustments in the single-rate model.
        
        Args:
            years_from_start: Number of years from forecast start
            total_forecast_years: Total years in forecast period
            
        Returns:
            Decay factor (0-1)
        """
        return self.calculate_exponential_decay(years_from_start, total_forecast_years)
    
    
    def calculate_custom_decay(
        self,
        years_from_start: int,
        decay_type: str = 'exponential',
        **kwargs
    ) -> float:
        """
        Calculate decay using custom decay functions.
        
        Args:
            years_from_start: Number of years from forecast start
            decay_type: Type of decay ('exponential', 'linear', 'logarithmic', 'sigmoid')
            **kwargs: Additional parameters for decay function
            
        Returns:
            Decay factor (0-1)
        """
        if decay_type == 'exponential':
            rate = kwargs.get('rate', 0.2)
            return np.exp(-rate * years_from_start)
        
        elif decay_type == 'linear':
            max_years = kwargs.get('max_years', 10)
            if years_from_start >= max_years:
                return 0.0
            return 1.0 - (years_from_start / max_years)
        
        elif decay_type == 'logarithmic':
            scale = kwargs.get('scale', 1.0)
            return 1.0 / (1.0 + scale * np.log(years_from_start + 1))
        
        elif decay_type == 'sigmoid':
            midpoint = kwargs.get('midpoint', 5)
            steepness = kwargs.get('steepness', 1.0)
            return 1.0 / (1.0 + np.exp(steepness * (years_from_start - midpoint)))
        
        else:
            raise ValueError(f"Unknown decay type: {decay_type}")
    
    def get_decay_profile(
        self,
        total_years: int,
        decay_type: str = 'exponential'
    ) -> dict:
        """
        Get decay profile for all forecast years.
        
        Args:
            total_years: Total forecast years
            decay_type: Type of decay profile to generate
            
        Returns:
            Dictionary mapping year index to decay factor
        """
        profile = {}
        
        for i in range(total_years):
            if decay_type == 'exponential':
                profile[i] = self.calculate_exponential_decay(i, total_years)
            else:
                profile[i] = self.calculate_custom_decay(i, decay_type)
        
        return profile