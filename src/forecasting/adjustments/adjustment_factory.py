"""
Factory for creating and orchestrating forecast adjustments.
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional
from .base import BaseAdjustment
from .news_adjustment import NewsAdjustment
from .indicator_adjustment import IndicatorAdjustment
from .fallback_strategies import FallbackStrategy, FallbackMethod


class AdjustmentFactory:
    """Factory for creating and applying forecast adjustments."""
    
    def __init__(self):
        """Initialize adjustment factory."""
        self.adjustments: List[BaseAdjustment] = []
        self.fallback_strategy = FallbackStrategy()
    
    def create_adjustments_from_config(self, config: Dict) -> List[BaseAdjustment]:
        """
        Create adjustment instances based on configuration.
        
        Args:
            config: Configuration dictionary with adjustment settings
            
        Returns:
            List of adjustment instances
        """
        adjustments = []
        weights = config.get('adjustment_weights', {})
        
        # Create news adjustment if enabled
        if config.get('use_news', False):
            news_adj = NewsAdjustment(
                weight=weights.get('news_weight', 0.7),
                confidence_multiplier=config.get('news_confidence_multiplier', 1.0)
            )
            adjustments.append(news_adj)
        
        # Create indicator adjustment if enabled
        if config.get('use_indicators', False):
            ind_adj = IndicatorAdjustment(
                weight=weights.get('indicator_weight', 0.3)
            )
            adjustments.append(ind_adj)
        
        self.adjustments = adjustments
        return adjustments
    
    def apply_all_adjustments(
        self,
        baseline_df: pd.DataFrame,
        news_data: Optional[pd.DataFrame],
        indicators_df: Optional[pd.DataFrame],
        config: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply all configured adjustments to baseline forecast.
        
        Args:
            baseline_df: Baseline forecast DataFrame
            news_data: News data for adjustment
            indicators_df: Indicator data for adjustment
            config: Configuration parameters
            
        Returns:
            Tuple of (adjusted DataFrame, combined details)
        """
        # Create adjustments if not already created
        if not self.adjustments:
            self.create_adjustments_from_config(config)
        
        # Start with baseline
        adjusted_df = baseline_df.copy()
        all_details = {
            'adjustments': [],
            'total_adjustment': 0.0
        }
        
        # Apply each adjustment
        for adjustment in self.adjustments:
            if isinstance(adjustment, NewsAdjustment) and news_data is not None:
                adjusted_df, details = adjustment.apply(adjusted_df, news_data, config)
                all_details['adjustments'].append(('news', details))
                all_details['total_adjustment'] += details.get('total_contribution', 0)
            
            elif isinstance(adjustment, IndicatorAdjustment) and indicators_df is not None:
                adjusted_df, details = adjustment.apply(adjusted_df, indicators_df, config)
                all_details['adjustments'].append(('indicators', details))
                all_details['total_adjustment'] += details.get('total_contribution', 0)
        
        return adjusted_df, all_details
    
    def apply_with_fallback(
        self,
        baseline_df: pd.DataFrame,
        country: str,
        country_news: Optional[pd.DataFrame],
        global_news: Optional[pd.DataFrame],
        indicators_df: Optional[pd.DataFrame],
        config: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply adjustments with fallback strategy for missing data.
        
        Args:
            baseline_df: Baseline forecast DataFrame
            country: Country name
            country_news: Country-specific news data
            global_news: Global news data
            indicators_df: Indicator data
            config: Configuration parameters
            
        Returns:
            Tuple of (adjusted DataFrame, details including fallback info)
        """
        # Determine if fallback is needed for news
        news_data = country_news
        fallback_info = None
        
        if (country_news is None or country_news.empty) and global_news is not None:
            # Apply fallback strategy
            news_data, confidence, reason = self.fallback_strategy.apply(
                country, global_news, config
            )
            fallback_info = {
                'used_fallback': True,
                'fallback_reason': reason,
                'confidence_multiplier': confidence
            }
            
            # Update config with fallback confidence
            config = config.copy()
            config['news_confidence_multiplier'] = confidence
        
        # Apply adjustments
        adjusted_df, details = self.apply_all_adjustments(
            baseline_df, news_data, indicators_df, config
        )
        
        # Add fallback info to details
        if fallback_info:
            details['fallback'] = fallback_info
        
        return adjusted_df, details
    
    @staticmethod
    def combine_adjustments(
        adjustments: List[Tuple[str, float]],
        combination_method: str = 'additive'
    ) -> float:
        """
        Combine multiple adjustment values.
        
        Args:
            adjustments: List of (name, value) tuples
            combination_method: How to combine ('additive', 'multiplicative', 'maximum')
            
        Returns:
            Combined adjustment value
        """
        if not adjustments:
            return 0.0
        
        values = [value for _, value in adjustments]
        
        if combination_method == 'additive':
            return sum(values)
        
        elif combination_method == 'multiplicative':
            result = 1.0
            for value in values:
                result *= (1 + value)
            return result - 1
        
        elif combination_method == 'maximum':
            return max(values, key=abs)
        
        else:
            raise ValueError(f"Unknown combination method: {combination_method}")