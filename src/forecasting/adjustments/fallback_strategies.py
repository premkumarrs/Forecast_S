"""
Fallback strategies for handling missing or insufficient data.
"""

import pandas as pd
from typing import Dict, Tuple, Optional
from enum import Enum


class FallbackMethod(Enum):
    """Enumeration of available fallback methods."""
    USE_GLOBAL_REDUCED = "use_global_reduced"
    SKIP_ADJUSTMENTS = "skip_adjustments"
    USE_REGIONAL = "use_regional"
    USE_SIMILAR_COUNTRIES = "use_similar_countries"
    USE_HISTORICAL = "use_historical"


class FallbackStrategy:
    """Manage fallback strategies for forecast adjustments."""
    
    def __init__(
        self,
        method: FallbackMethod = FallbackMethod.USE_GLOBAL_REDUCED,
        confidence_multiplier: float = 0.5
    ):
        """
        Initialize fallback strategy.
        
        Args:
            method: Fallback method to use
            confidence_multiplier: Confidence reduction factor
        """
        self.method = method
        self.confidence_multiplier = confidence_multiplier
        self.fallback_reason = ""
    
    def apply(
        self,
        country: str,
        global_data: pd.DataFrame,
        config: Dict,
        regional_data: Optional[pd.DataFrame] = None,
        historical_data: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, float, str]:
        """
        Apply fallback strategy when country-specific data is insufficient.
        
        Args:
            country: Country name
            global_data: Global news/indicator data
            config: Configuration parameters
            regional_data: Optional regional data
            historical_data: Optional historical data
            
        Returns:
            Tuple of (fallback data, confidence multiplier, reason)
        """
        # Get method from config or use default
        config_fallback = config.get('fallback_strategy', {})
        method_str = config_fallback.get('method', self.method.value)
        confidence = config_fallback.get('confidence_multiplier', self.confidence_multiplier)
        
        # Convert string to enum if needed
        if isinstance(method_str, str):
            try:
                method = FallbackMethod(method_str)
            except ValueError:
                method = self.method
        else:
            method = method_str
        
        # Apply appropriate fallback
        if method == FallbackMethod.USE_GLOBAL_REDUCED:
            return self._use_global_reduced(global_data, confidence, country)
        
        elif method == FallbackMethod.SKIP_ADJUSTMENTS:
            return self._skip_adjustments(country)
        
        elif method == FallbackMethod.USE_REGIONAL:
            return self._use_regional(regional_data, global_data, confidence, country)
        
        elif method == FallbackMethod.USE_SIMILAR_COUNTRIES:
            return self._use_similar_countries(country, global_data, confidence)
        
        elif method == FallbackMethod.USE_HISTORICAL:
            return self._use_historical(historical_data, global_data, confidence, country)
        
        else:
            # Default fallback
            return self._use_global_reduced(global_data, 0.5, country)
    
    def _use_global_reduced(
        self,
        global_data: pd.DataFrame,
        confidence: float,
        country: str
    ) -> Tuple[pd.DataFrame, float, str]:
        """Use global data with reduced confidence."""
        reason = f"Using global news at {confidence:.0%} confidence (insufficient {country} data)"
        return global_data, confidence, reason
    
    def _skip_adjustments(self, country: str) -> Tuple[pd.DataFrame, float, str]:
        """Skip all adjustments."""
        reason = f"Skipping news adjustments for {country} (insufficient data)"
        return pd.DataFrame(), 0.0, reason
    
    def _use_regional(
        self,
        regional_data: Optional[pd.DataFrame],
        global_data: pd.DataFrame,
        confidence: float,
        country: str
    ) -> Tuple[pd.DataFrame, float, str]:
        """Use regional data if available."""
        if regional_data is not None and not regional_data.empty:
            reason = f"Using regional data at {confidence:.0%} confidence for {country}"
            return regional_data, confidence, reason
        else:
            return self._use_global_reduced(global_data, confidence * 0.8, country)
    
    def _use_similar_countries(
        self,
        country: str,
        global_data: pd.DataFrame,
        confidence: float
    ) -> Tuple[pd.DataFrame, float, str]:
        """Use data from similar countries (placeholder for more complex logic)."""
        # This would require country similarity mapping
        # For now, fall back to global with reduced confidence
        reason = f"Using similar country patterns at {confidence * 0.7:.0%} confidence for {country}"
        return global_data, confidence * 0.7, reason
    
    def _use_historical(
        self,
        historical_data: Optional[pd.DataFrame],
        global_data: pd.DataFrame,
        confidence: float,
        country: str
    ) -> Tuple[pd.DataFrame, float, str]:
        """Use historical patterns if available."""
        if historical_data is not None and not historical_data.empty:
            reason = f"Using historical patterns at {confidence:.0%} confidence for {country}"
            return historical_data, confidence, reason
        else:
            return self._use_global_reduced(global_data, confidence * 0.6, country)
    
    @staticmethod
    def determine_best_fallback(
        country: str,
        available_data: Dict[str, pd.DataFrame]
    ) -> FallbackMethod:
        """
        Determine the best fallback method based on available data.
        
        Args:
            country: Country name
            available_data: Dictionary of available data types
            
        Returns:
            Recommended fallback method
        """
        # Check what data is available
        has_regional = 'regional' in available_data and not available_data['regional'].empty
        has_historical = 'historical' in available_data and not available_data['historical'].empty
        has_global = 'global' in available_data and not available_data['global'].empty
        
        # Determine best fallback
        if has_regional:
            return FallbackMethod.USE_REGIONAL
        elif has_historical:
            return FallbackMethod.USE_HISTORICAL
        elif has_global:
            return FallbackMethod.USE_GLOBAL_REDUCED
        else:
            return FallbackMethod.SKIP_ADJUSTMENTS
    
    def get_confidence_for_data_quality(
        self,
        data: pd.DataFrame,
        min_records: int = 10
    ) -> float:
        """
        Calculate confidence multiplier based on data quality.
        
        Args:
            data: Input data
            min_records: Minimum records for full confidence
            
        Returns:
            Confidence multiplier (0-1)
        """
        if data is None or data.empty:
            return 0.0
        
        record_count = len(data)
        
        if record_count >= min_records:
            return 1.0
        else:
            # Linear scaling based on record count
            return record_count / min_records