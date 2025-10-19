"""
Indicator-based forecast adjustments.
"""

import pandas as pd
from typing import Dict, Tuple, Optional
from .base import BaseAdjustment


class IndicatorAdjustment(BaseAdjustment):
    """Calculate and apply indicator-based adjustments to forecasts."""
    
    def __init__(self, weight: float = 0.3):
        """
        Initialize indicator adjustment.
        
        Args:
            weight: Weight of indicator adjustment (0-1)
        """
        super().__init__(weight)
        self.indicator_signals = {}
    
    def calculate(self, data: pd.DataFrame, config: Dict) -> float:
        """
        Calculate indicator-based adjustment from indicator data.
        
        Args:
            data: Indicator data with value and indicator columns
            config: Configuration with indicator_weights
            
        Returns:
            Average indicator adjustment value
        """
        if data is None or data.empty:
            return 0.0
        
        ind_weights = config.get('indicator_weights', {})
        if not ind_weights:
            return 0.0
        
        # Determine indicator column name
        indicator_col = (
            'indicator_key' if 'indicator_key' in data.columns
            else 'indicator'
        )
        
        # Calculate YoY growth for indicators
        data = data.copy()
        data['yoy_growth'] = data.groupby(indicator_col)['value'].pct_change()
        
        # Calculate yearly signals
        yearly_signals = []
        for year, group in data.groupby('year'):
            signal = 0.0
            year_indicators = {}
            
            for ind_name, ind_weight in ind_weights.items():
                if ind_name in group[indicator_col].values:
                    growth = group[group[indicator_col] == ind_name]['yoy_growth'].iloc[0]
                    if pd.notna(growth):
                        weighted_growth = growth * ind_weight
                        signal += weighted_growth
                        year_indicators[ind_name] = {
                            'growth': growth * 100,
                            'weight': ind_weight,
                            'contribution': weighted_growth * 100
                        }
            
            yearly_signals.append(signal)
            self.indicator_signals[year] = year_indicators
        
        # Calculate average adjustment
        if yearly_signals:
            self.adjustment_value = sum(yearly_signals) / len(yearly_signals)
        else:
            self.adjustment_value = 0.0
        
        # Validate adjustment bounds
        self.adjustment_value = self.validate_adjustment(
            self.adjustment_value, -0.3, 0.3
        )
        
        return self.adjustment_value
    
    def apply(
        self,
        baseline_df: pd.DataFrame,
        data: pd.DataFrame,
        config: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply indicator adjustment to baseline forecast.
        
        Args:
            baseline_df: Baseline forecast DataFrame
            data: Indicator data
            config: Configuration parameters
            
        Returns:
            Tuple of (adjusted DataFrame, adjustment details)
        """
        # Calculate base adjustment
        self.calculate(data, config)
        
        # Apply to forecast (no decay for indicators)
        adjusted_df = baseline_df.copy()
        forecast_mask = adjusted_df['type'] == 'Forecast'
        forecast_years = adjusted_df[forecast_mask]['year'].values
        
        year_adjustments = {}
        
        for year in forecast_years:
            year_mask = (adjusted_df['year'] == year) & forecast_mask
            
            # Apply weighted adjustment
            total_adj = self.get_weighted_adjustment(self.adjustment_value)
            
            # Apply to forecast
            adjustment_factor = 1 + total_adj
            adjusted_df.loc[year_mask, 'value_hat'] *= adjustment_factor
            
            # Store details
            year_adjustments[year] = {
                'indicator_pct': total_adj * 100,
                'indicators': self.indicator_signals.get(year, {})
            }
        
        # Prepare details
        self.details = {
            'average_adjustment': self.adjustment_value * 100,
            'year_adjustments': year_adjustments,
            'total_contribution': self.get_weighted_adjustment(self.adjustment_value) * 100
        }
        
        return adjusted_df, self.get_details()
    
    def calculate_momentum(
        self,
        data: pd.DataFrame,
        lookback_years: int = 3
    ) -> float:
        """
        Calculate indicator momentum over recent years.
        
        Args:
            data: Indicator data
            lookback_years: Number of years to consider
            
        Returns:
            Momentum value (-1 to 1)
        """
        if data is None or data.empty:
            return 0.0
        
        # Get recent years
        recent_data = data.nlargest(lookback_years, 'year')
        
        if len(recent_data) < 2:
            return 0.0
        
        # Calculate trend
        years = recent_data['year'].values
        values = recent_data['value'].values
        
        # Simple linear regression for trend
        from scipy import stats
        slope, _, r_value, _, _ = stats.linregress(years, values)
        
        # Normalize to -1 to 1 range
        momentum = max(-1, min(1, slope / values.mean() if values.mean() != 0 else 0))
        
        return momentum