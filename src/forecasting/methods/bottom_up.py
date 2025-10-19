"""
Bottom-up forecast method.
Pipeline: Country forecasts -> Regional Aggregation -> Worldwide Aggregation
"""

import pandas as pd
import logging
from typing import Dict, Optional, List

from ..models.baseline_factory import BaselineModelFactory
from ..adjustments import calculate_unified_adjustment, apply_fallback_strategy
from ...regions import aggregate_countries_to_regions
from ...constants import ForecastMode
from .existing_mode_helpers import (
    build_baseline_dataframe,
    apply_adjustment_window,
    derive_hist_cutoff,
)

logger = logging.getLogger(__name__)

def forecast_bottom_up(
    unified_data: Dict,
    config: Dict,
    analyzed_news: Optional[pd.DataFrame] = None,
    country_news: Optional[Dict[str, pd.DataFrame]] = None
) -> Dict:
    """
    Bottom-up: Individual country forecasts aggregated to regions and then to global.

    Args:
        unified_data: Unified data structure
        config: Configuration dictionary
        analyzed_news: Global news (fallback)
        country_news: Country-specific news
    
    Returns:
        Dictionary with global, regional, and country forecasts
    """
    logger.info("Starting Bottom-Up forecast method")

    if config.get('forecast_mode', ForecastMode.CLASSIC.value) == ForecastMode.EXISTING_FORECAST_NEWS.value:
        return _forecast_existing_news_bottom_up(
            unified_data,
            config,
            analyzed_news,
            country_news
        )

    country_market = unified_data.get('market_value', {}).get('country', pd.DataFrame())
    country_indicators = unified_data.get('indicators', {}).get('country', pd.DataFrame())
    
    if country_market.empty:
        return {'error': 'No country market data available for bottom-up forecast'}
    
    countries = country_market['country'].unique()
    logger.info(f"Processing {len(countries)} countries.")
    
    country_forecasts = {}
    adjustments_applied = set()
    countries_with_news = []
    countries_indicators_only = []
    countries_insufficient_data = []
    all_adjustment_details = {}
    
    # Step 1: Generate forecast for each country
    baseline_country_forecasts = {}  # Store baseline forecasts separately
    for country in countries:
        country_data = country_market[country_market['country'] == country]
        
        if len(country_data) < 3:
            countries_insufficient_data.append(country)
            continue
            
        baseline_forecast = BaselineModelFactory.generate_baseline_forecast(
            country_data,
            config.get('forecast_method', '3-yr CAGR'),
            config['hist_cutoff'],
            config['forecast_until']
        )
        
        # Store baseline for later reference
        baseline_country_forecasts[country] = baseline_forecast.copy()
        
        # Get country-specific news and indicators
        country_news_df = country_news.get(country) if country_news is not None else None
        if country_news_df is None:
            country_news_df = pd.DataFrame()
        news_confidence = 1.0

        # Apply fallback if country news is insufficient
        fallback_reason = None
        if country_news_df.empty or len(country_news_df) < 5:
            country_news_df, news_confidence, fallback_reason = apply_fallback_strategy(
                country,
                analyzed_news, # Global news for fallback
                config
            )

        country_indicators_df = country_indicators[country_indicators['country'] == country] if not country_indicators.empty else None

        # Apply unified adjustment for the country
        # JIT override via calibration (country → fallback to global)
        cfg = dict(config)
        cal_scope = {}
        try:
            cal_scope = getattr(country_news_df, 'attrs', {}).get('calibration', {}) or {}
            if not cal_scope and isinstance(analyzed_news, pd.DataFrame):
                cal_scope = getattr(analyzed_news, 'attrs', {}).get('calibration', {}) or {}
        except Exception:
            cal_scope = {}
        if cal_scope:
            aw = dict(cfg.get('adjustment_weights', {}))
            cfg['long_term_decay_rate'] = cal_scope.get('long_term_decay_rate', cfg.get('long_term_decay_rate', 0.60))
            cfg['adjustment_weights'] = aw

        adjusted_country_forecast, adjustment_details = calculate_unified_adjustment(
            baseline_forecast,
            country_news_df,
            country_indicators_df,
            cfg,
            news_confidence_multiplier=news_confidence
        )

        # Add fallback info to details
        if fallback_reason:
            adjustment_details['fallback_used'] = True
            adjustment_details['fallback_reason'] = fallback_reason
        else:
            adjustment_details['fallback_used'] = False

        country_forecasts[country] = adjusted_country_forecast
        all_adjustment_details[country] = adjustment_details

        if adjustment_details.get('news_contribution', 0) != 0:
            countries_with_news.append(country)
            adjustments_applied.add('country_news')
        else:
            countries_indicators_only.append(country)

        if adjustment_details.get('indicator_contribution', 0) != 0:
            adjustments_applied.add('indicators')
    
    if not country_forecasts:
        return {'error': 'No countries have sufficient data for forecasting'}
    
    # Step 2: Aggregate countries to regions
    logger.info("Aggregating country forecasts to regional forecasts...")
    region_forecasts = aggregate_countries_to_regions(country_forecasts)
    
    # Step 3: Extract the final 'Worldwide' forecast from regional forecasts
    global_forecast = region_forecasts.get('Worldwide')
    if global_forecast is None:
        logger.warning("'Worldwide' region not found in aggregation. Manually aggregating all countries as fallback.")
        global_forecast = _manual_aggregate(list(country_forecasts.values()))

    logger.info(f"Bottom-Up forecast complete.")
    
    return {
        'global_forecast': global_forecast,
        'region_forecasts': region_forecasts,
        'country_forecasts': country_forecasts,
        'baseline_country_forecasts': baseline_country_forecasts,  # Add baseline forecasts
        'method': 'bottom_up',
        'adjustments_applied': list(adjustments_applied),
        'adjustment_details': all_adjustment_details, # NEW
        'metadata': {
            'countries_forecasted': len(country_forecasts),
            'regions_forecasted': len(region_forecasts),
            'countries_with_news': countries_with_news,
            'countries_indicators_only': countries_indicators_only,
            'countries_insufficient_data': countries_insufficient_data,
            'forecast_years': list(range(config['hist_cutoff'] + 1, config['forecast_until'] + 1)),
            'adjustment_breakdown': all_adjustment_details # NEW
        }
    }


def _forecast_existing_news_bottom_up(
    unified_data: Dict,
    config: Dict,
    analyzed_news: Optional[pd.DataFrame],
    country_news: Optional[Dict[str, pd.DataFrame]]
) -> Dict:
    country_market = unified_data.get('market_value', {}).get('country', pd.DataFrame())
    if country_market is None or country_market.empty:
        return {'error': 'No country market data available for bottom-up forecast'}

    forecast_until = config.get('forecast_until')
    if forecast_until is None:
        return {'error': 'forecast_until is required for existing forecasts mode'}

    start_year = config.get('adjustment_start_year')
    end_year = config.get('adjustment_end_year')
    derived_hist_cutoff = derive_hist_cutoff(config)

    countries = country_market['country'].unique()
    country_forecasts = {}
    baseline_country_forecasts = {}
    adjustments_applied = set()
    countries_with_news = []
    countries_indicators_only = []
    countries_insufficient_data = []
    all_adjustment_details = {}

    cfg = dict(config)
    cfg['hist_cutoff'] = derived_hist_cutoff
    cfg['use_indicators'] = False
    cfg['use_news'] = True

    for country in countries:
        country_data = country_market[country_market['country'] == country]
        if country_data.empty:
            continue

        baseline_forecast = build_baseline_dataframe(
            country_data,
            forecast_until,
            derived_hist_cutoff
        )

        if baseline_forecast.empty:
            countries_insufficient_data.append(country)
            continue

        baseline_country_forecasts[country] = baseline_forecast.copy()

        country_news_df = (country_news or {}).get(country) if country_news else None
        if country_news_df is None:
            country_news_df = pd.DataFrame()
        news_confidence = 1.0
        fallback_reason = None

        if country_news_df.empty or len(country_news_df) < 5:
            country_news_df, news_confidence, fallback_reason = apply_fallback_strategy(
                country,
                analyzed_news,
                cfg
            )

        adjusted_country_forecast, adjustment_details = calculate_unified_adjustment(
            baseline_forecast,
            country_news_df,
            None,
            cfg,
            news_confidence_multiplier=news_confidence
        )

        adjusted_country_forecast, adjustment_details = apply_adjustment_window(
            adjusted_country_forecast,
            baseline_forecast,
            adjustment_details,
            start_year,
            end_year
        )

        if adjustment_details and adjustment_details.get('news_contribution', 0) != 0:
            countries_with_news.append(country)
            adjustments_applied.add('country_news')
        else:
            countries_indicators_only.append(country)

        if fallback_reason:
            adjustment_details = dict(adjustment_details or {})
            adjustment_details['fallback_used'] = True
            adjustment_details['fallback_reason'] = fallback_reason
        else:
            adjustment_details = dict(adjustment_details or {})
            adjustment_details['fallback_used'] = False

        all_adjustment_details[country] = adjustment_details
        country_forecasts[country] = adjusted_country_forecast


    if not country_forecasts:
        return {'error': 'No countries have sufficient data for forecasting'}

    region_forecasts = aggregate_countries_to_regions(country_forecasts)
    global_forecast = region_forecasts.get('Worldwide')
    if global_forecast is None:
        global_forecast = _manual_aggregate(list(country_forecasts.values()))

    metadata = {
        'countries_forecasted': len(country_forecasts),
        'regions_forecasted': len(region_forecasts),
        'countries_with_news': countries_with_news,
        'countries_indicators_only': countries_indicators_only,
        'countries_insufficient_data': countries_insufficient_data,
        'forecast_years': list(range((derived_hist_cutoff or 0) + 1, int(forecast_until) + 1)),
        'adjustment_breakdown': all_adjustment_details
    }

    return {
        'global_forecast': global_forecast,
        'region_forecasts': region_forecasts,
        'country_forecasts': country_forecasts,
        'baseline_country_forecasts': baseline_country_forecasts,
        'method': 'bottom_up',
        'adjustments_applied': list(adjustments_applied),
        'adjustment_details': all_adjustment_details,
        'metadata': metadata
    }

def _manual_aggregate(forecast_dfs: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """A simple fallback to sum a list of forecast dataframes."""
    if not forecast_dfs:
        return None
    
    total_df = forecast_dfs[0].copy()
    total_df['value_hat'] = 0
    
    for df in forecast_dfs:
        total_df['value_hat'] += df['value_hat']
        
    return total_df
