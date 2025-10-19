"""
News-based forecast adjustments.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from .base import BaseAdjustment
from .temporal_decay import TemporalDecay


class NewsAdjustment(BaseAdjustment):
    """Calculate and apply news-based adjustments to forecasts."""
    
    def __init__(
        self,
        weight: float = 0.7,
        confidence_multiplier: float = 1.0
    ):
        """
        Initialize news adjustment.
        
        Args:
            weight: Weight of news adjustment (0-1)
            confidence_multiplier: Confidence in news data (0-1)
        """
        super().__init__(weight)
        self.confidence_multiplier = confidence_multiplier
        self.adjustment_value = 0.0
        self.category_breakdown = {}
        self.temporal_decay = TemporalDecay()
    
    def _calculate_legacy(self, data: pd.DataFrame, config: Dict) -> float:
        """Legacy calculation method for backward compatibility."""
        # Filter relevant & valid
        if data is None or data.empty:
            return 0.0
        
        df = data.copy()
        if 'relevant' in df.columns:
            df = df[df['relevant'] == 1]
        
        # Filter for relevant news (non-zero growth rate)
        df = df[df['growth_rate'] != 0]
        
        if df.empty:
            return 0.0
        
        # Single recency half-life (days)
        hl = int(config.get('news_half_life_days', 90))  # new single setting
        
        # Single recency-weighted average (percentage -> decimal)
        news_avg_pct = self._recency_weighted_mean(df, hl)  # returns % value
        if pd.isna(news_avg_pct):
            news_avg_pct = df.get('growth_rate', pd.Series([0.0])).mean()
        
        adjustment_value = (news_avg_pct / 100.0) * self.confidence_multiplier
        
        # Bound to sane limits
        return self.validate_adjustment(adjustment_value, -0.3, 0.3)
    
    def _recency_weighted_mean(self, df: pd.DataFrame, half_life_days: int) -> float:
        """
        Calculate recency-weighted mean of growth rates.
        
        Args:
            df: DataFrame with 'growth_rate' and 'date' columns
            half_life_days: Half-life period in days for exponential decay
            
        Returns:
            Weighted mean growth rate
        """
        # Defensive: handle numeric and date conversions
        x = pd.to_numeric(df.get('growth_rate'), errors='coerce')
        d = pd.to_datetime(df.get('date'), errors='coerce')
        
        # Filter valid data
        m = x.notna() & d.notna()
        if not m.any():
            return float('nan')
        
        # Calculate age in days from current time
        # Ensure timezone consistency
        current_time = pd.Timestamp.utcnow()
        dates = d[m]
        if dates.dt.tz is not None:
            # Convert timezone-aware to naive
            dates = dates.dt.tz_localize(None)
        if current_time.tz is not None:
            current_time = current_time.tz_localize(None)
        
        age_days = (current_time - dates).dt.days.clip(lower=0)
        
        # Calculate weights using half-life formula: weight(t) = 0.5 ** (age_days / half_life)
        # Ensure half_life is positive to avoid division issues
        half_life_days = max(1, abs(half_life_days))
        w = np.power(0.5, age_days / half_life_days)
        
        return float(np.average(x[m], weights=w))
    
    def calculate(self, data: pd.DataFrame, config: Dict) -> float:
        """
        Calculate news-based adjustment using category-specific MA windows.
        
        Args:
            data: News data with growth_rate columns
            config: Configuration parameters
            
        Returns:
            Total news adjustment value
        """
        if data is None or data.empty:
            self.adjustment_value = 0.0
            self.category_breakdown = {}
            return self.adjustment_value
        
        # Load categories with MA windows
        categories = config.get('categories', [])
        
        if not categories:
            # Fallback to old single average
            self.adjustment_value = self._calculate_legacy(data, config)
            self.category_breakdown = {}
            return self.adjustment_value
        
        # Filter relevant news first
        df = data.copy()
        if 'relevant' in df.columns:
            df = df[df['relevant'] == 1]
        df = df[df['growth_rate'] != 0]
        
        if df.empty:
            self.adjustment_value = 0.0
            self.category_breakdown = {}
            return self.adjustment_value
        
        # Group news by category and date
        category_impacts = []
        self.category_breakdown = {}
        
        for category in categories:
            cat_name = category.get('name', 'Unknown')
            ma_window = category.get('ma_window_days', 30)
            
            if ma_window == 0:
                # Neutral category - skip
                self.category_breakdown[cat_name] = {
                    'window': 0,
                    'days_with_data': 0,
                    'coverage_pct': 0,
                    'ma_score': 0.0,
                    'status': 'Neutral (not contributing)',
                    'contribution': 0.0
                }
                continue
            
            # Filter news for this category
            if 'category' in df.columns:
                cat_data = df[df['category'] == cat_name]
            else:
                # If no category column, use all data for first category only
                cat_data = df if len(category_impacts) == 0 else pd.DataFrame()
            
            if cat_data.empty:
                self.category_breakdown[cat_name] = {
                    'window': ma_window,
                    'days_with_data': 0,
                    'coverage_pct': 0,
                    'ma_score': 0.0,
                    'status': 'No recent data',
                    'contribution': 0.0
                }
                continue
            
            # Calculate MA for this category
            cutoff_date = pd.Timestamp.utcnow() - pd.Timedelta(days=ma_window)
            # Convert dates and ensure timezone consistency
            cat_data = cat_data.copy()  # Avoid SettingWithCopyWarning
            cat_data['date'] = pd.to_datetime(cat_data['date'])
            # Make dates timezone-naive for comparison
            if cat_data['date'].dt.tz is not None:
                cat_data['date'] = cat_data['date'].dt.tz_localize(None)
            cutoff_date_naive = cutoff_date.tz_localize(None) if cutoff_date.tz else cutoff_date
            recent_data = cat_data[cat_data['date'] >= cutoff_date_naive]
            
            if recent_data.empty:
                self.category_breakdown[cat_name] = {
                    'window': ma_window,
                    'days_with_data': 0,
                    'coverage_pct': 0,
                    'ma_score': 0.0,
                    'status': 'No data in window',
                    'contribution': 0.0
                }
                continue
            
            # Apply recency weighting within the MA window
            # Calculate weights based on days ago from today
            recent_data['days_ago'] = (pd.Timestamp.utcnow().tz_localize(None) - recent_data['date']).dt.days
            
            # Use exponential decay for recency (half-life = 1/3 of window)
            half_life = max(ma_window / 3, 7)  # At least 7 days half-life
            recent_data['recency_weight'] = np.exp(-0.693 * recent_data['days_ago'] / half_life)
            
            # Calculate weighted average by day, then overall MA
            daily_weighted = recent_data.groupby(recent_data['date'].dt.date).apply(
                lambda x: np.average(x['growth_rate'], weights=x['recency_weight']) if len(x) > 0 else 0
            )
            ma_value = daily_weighted.mean() / 100.0  # Convert to decimal
            
            days_with_data = len(daily_weighted)
            coverage_pct = (days_with_data / ma_window) * 100 if ma_window > 0 else 100
            
            category_impacts.append(ma_value)
            self.category_breakdown[cat_name] = {
                'window': ma_window,
                'days_with_data': days_with_data,
                'coverage_pct': coverage_pct,
                'ma_score': ma_value * 100,
                'status': 'Active',
                'contribution': ma_value
            }
        
        # Equal weighting across active categories
        if category_impacts:
            self.adjustment_value = np.mean(category_impacts) * self.confidence_multiplier
        else:
            self.adjustment_value = 0.0
        
        # Bound to sane limits
        self.adjustment_value = self.validate_adjustment(self.adjustment_value, -0.3, 0.3)
        
        return self.adjustment_value
    
    def apply(
        self,
        baseline_df: pd.DataFrame,
        data: pd.DataFrame,
        config: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply news adjustment to baseline forecast with temporal decay.
        
        Args:
            baseline_df: Baseline forecast DataFrame
            data: News data
            config: Configuration parameters
            
        Returns:
            Tuple of (adjusted DataFrame, adjustment details)
        """
        # Compute single avg
        news_avg = self.calculate(data, config)
        
        adjusted_df = baseline_df.copy()
        forecast_mask = adjusted_df['type'] == 'Forecast'
        forecast_years = adjusted_df[forecast_mask]['year'].values
        
        # One decay function over horizon
        temporal_decay = TemporalDecay(
            short_term_horizon=None,  # not used now
            long_term_decay_rate=float(config.get('long_term_decay_rate', 0.6))
        )
        
        weights = config.get('adjustment_weights', {})
        news_weight = float(weights.get('news_weight', 0.7))
        
        year_adjustments = {}
        total_years = len(forecast_years)
        
        for i, year in enumerate(forecast_years):
            decay = temporal_decay.calculate_news_decay(i, total_years)  # single curve using exponential decay
            
            # NEWS branch = News_Weight × news_avg × Temporal_Decay(year)
            news_term = news_weight * news_avg * decay
            
            # Apply only the news branch here; the indicator branch is added in unified flow
            adj_factor = 1.0 + news_term
            
            year_mask = (adjusted_df['year'] == year) & forecast_mask
            adjusted_df.loc[year_mask, 'value_hat'] *= adj_factor
            
            year_adjustments[year] = {
                'news_pct': news_term * 100.0,
                'news_avg_pct': news_avg * 100.0,
                'decay': decay,
            }
        
        self.details = {
            'news_adjustment': news_avg * 100.0,
            'confidence_multiplier': self.confidence_multiplier,
            'year_adjustments': year_adjustments,
            'total_contribution': float(np.mean([v['news_pct'] for v in year_adjustments.values()])) if year_adjustments else 0.0,
        }
        
        # Add category breakdown if available
        if hasattr(self, 'category_breakdown'):
            self.details['category_breakdown'] = self.category_breakdown
        
        return adjusted_df, self.get_details()
    
    def get_details(self) -> Dict:
        """
        Get adjustment details including category breakdown.
        
        Returns:
            Dictionary with adjustment details including category breakdown
        """
        details = super().get_details()
        
        # Add details set by apply method if available
        if hasattr(self, 'details') and self.details:
            details.update(self.details)
        
        # Always include category breakdown if available
        if hasattr(self, 'category_breakdown'):
            details['category_breakdown'] = self.category_breakdown
            
        return details
    
    def apply_with_confidence(
        self,
        baseline_df: pd.DataFrame,
        data: pd.DataFrame,
        config: Dict,
        confidence: float
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply news adjustment with specific confidence level.
        
        Args:
            baseline_df: Baseline forecast DataFrame
            data: News data
            config: Configuration parameters
            confidence: Confidence multiplier for this application
            
        Returns:
            Tuple of (adjusted DataFrame, adjustment details)
        """
        original_confidence = self.confidence_multiplier
        self.confidence_multiplier = confidence
        result = self.apply(baseline_df, data, config)
        self.confidence_multiplier = original_confidence
        return result