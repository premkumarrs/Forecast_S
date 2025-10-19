"""
Country-specific forecast method.
Pipeline: Selected countries → Individual detailed forecasts
"""

import pandas as pd
import logging
from typing import Dict, Optional, List

from ..models.baseline_factory import BaselineModelFactory
from ..adjustments import calculate_unified_adjustment, apply_fallback_strategy
from ...constants import ForecastMode
from .existing_mode_helpers import (
    build_baseline_dataframe,
    apply_adjustment_window,
    derive_hist_cutoff,
)

logger = logging.getLogger(__name__)


def forecast_country_specific(
    unified_data: Dict,
    config: Dict,
    selected_countries: List[str],
    analyzed_news: Optional[pd.DataFrame] = None,
    country_news: Optional[Dict[str, pd.DataFrame]] = None
) -> Dict:
    """
    Country-specific: Deep dive into selected countries only.
    
    Args:
        unified_data: Unified data structure
        config: Configuration dictionary
        selected_countries: List of countries to forecast
        analyzed_news: Global news (fallback)
        country_news: Country-specific news
    
    Returns:
        Dictionary with country forecasts (no global aggregation)
    """
    logger.info("Starting Country-Specific forecast method")
    logger.info(f"Selected countries: {selected_countries}")
    logger.info(f"Country news type received: {type(country_news)}")

    if config.get('forecast_mode', ForecastMode.CLASSIC.value) == ForecastMode.EXISTING_FORECAST_NEWS.value:
        return _forecast_existing_news_country_specific(
            unified_data,
            config,
            selected_countries,
            analyzed_news,
            country_news
        )
    
    # Extract data
    country_market = unified_data.get('market_value', {}).get('country', pd.DataFrame())
    country_indicators = unified_data.get('indicators', {}).get('country', pd.DataFrame())
    
    if country_market.empty:
        return {'error': 'No country market data available'}
    
    if not selected_countries:
        return {'error': 'No countries selected for country-specific forecast'}
    
    available_countries = country_market['country'].unique()
    country_forecasts = {}
    adjustments_applied = set()
    countries_with_insufficient_data = []
    countries_with_news = []
    countries_indicators_only = []
    all_adjustment_details = {}  # Track adjustment details for each country
    
    # Step 1: Generate detailed forecast for each selected country
    for country in selected_countries:
        logger.info(f"Processing selected country: {country}")
        
        if country not in available_countries:
            countries_with_insufficient_data.append(f"{country}: not in data")
            logger.warning(f"Country {country} not found in available data")
            continue
            
        country_data = country_market[country_market['country'] == country]
        
        if len(country_data) < 3:  # Need minimum data points
            countries_with_insufficient_data.append(f"{country}: insufficient data points ({len(country_data)})")
            logger.warning(f"Skipping {country}: insufficient data points ({len(country_data)})")
            continue
        
        # Generate baseline forecast
        baseline_forecast = BaselineModelFactory.generate_baseline_forecast(
            country_data,
            config.get('forecast_method', '3-yr CAGR'),
            config['hist_cutoff'],
            config['forecast_until']
        )
        
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

        # Apply unified adjustment for the country with JIT calibration override
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

        adjusted_forecast, adjustment_details = calculate_unified_adjustment(
            baseline_forecast,
            country_news_df,
            country_indicators_df,
            cfg,
            news_confidence_multiplier=news_confidence
        )

        if adjustment_details.get('news_contribution', 0) != 0:
            countries_with_news.append(country)
            adjustments_applied.add('country_news')
        else:
            countries_indicators_only.append(country)

        if adjustment_details.get('indicator_contribution', 0) != 0:
            adjustments_applied.add('indicators')
        
        # Add fallback info to details
        if fallback_reason:
            adjustment_details['fallback_used'] = True
            adjustment_details['fallback_reason'] = fallback_reason
        else:
            adjustment_details['fallback_used'] = False

        # Store adjustment details at top level for UI access
        all_adjustment_details[country] = adjustment_details
        
        # Store detailed metadata for each country
        country_metadata = {
            'data_points': len(country_data),
            'has_indicators': not country_indicators.empty and not country_indicators[country_indicators['country'] == country].empty,
            'has_country_news': country_news_df is not None and not country_news_df.empty,
            'news_articles_count': len(country_news_df) if country_news_df is not None else 0,
            'adjustments_applied': list(adjustments_applied),
            'adjustment_details': adjustment_details,
            'fallback_used': fallback_reason is not None,
            'fallback_reason': fallback_reason
        }
        
        country_forecasts[country] = {
            'forecast': adjusted_forecast,
            'baseline': baseline_forecast,
            'metadata': country_metadata
        }
    
    if not country_forecasts:
        error_details = "; ".join(countries_with_insufficient_data)
        logger.error(f"No countries have sufficient data for forecasting: {error_details}")
        return {'error': f'No countries have sufficient data for forecasting. Details: {error_details}'}
    
    logger.info(f"Country-Specific forecast complete: {len(countries_with_news)} countries with news, {len(countries_indicators_only)} indicators-only")
    
    return {
        'country_forecasts': country_forecasts,
        'method': 'country_specific',
        'adjustments_applied': list(adjustments_applied),
        'adjustment_details': all_adjustment_details,  # Add this for UI access
        'metadata': {
            'countries_forecasted': len(country_forecasts),
            'countries_requested': len(selected_countries),
            'countries_with_news': countries_with_news,
            'countries_indicators_only': countries_indicators_only,
            'countries_with_insufficient_data': countries_with_insufficient_data,
            'news_coverage': f"{len(countries_with_news)}/{len(country_forecasts)} countries",
            'forecast_years': list(range(config['hist_cutoff'] + 1, config['forecast_until'] + 1)),
            'adjustment_breakdown': all_adjustment_details  # Also add here for consistency
        }
    }


def _forecast_existing_news_country_specific(
    unified_data: Dict,
    config: Dict,
    selected_countries: List[str],
    analyzed_news: Optional[pd.DataFrame],
    country_news: Optional[Dict[str, pd.DataFrame]]
) -> Dict:
    country_market = unified_data.get('market_value', {}).get('country', pd.DataFrame())
    if country_market is None or country_market.empty:
        return {'error': 'No country market data available'}

    if not selected_countries:
        return {'error': 'No countries selected for country-specific forecast'}

    forecast_until = config.get('forecast_until')
    if forecast_until is None:
        return {'error': 'forecast_until is required for existing forecasts mode'}

    start_year = config.get('adjustment_start_year')
    end_year = config.get('adjustment_end_year')
    derived_hist_cutoff = derive_hist_cutoff(config)

    available_countries = set(country_market['country'].unique())
    country_forecasts = {}
    adjustments_applied = set()
    countries_with_insufficient_data = []
    countries_with_news = []
    countries_indicators_only = []
    all_adjustment_details = {}

    cfg = dict(config)
    cfg['hist_cutoff'] = derived_hist_cutoff
    cfg['use_indicators'] = False
    cfg['use_news'] = True

    for country in selected_countries:
        if country not in available_countries:
            countries_with_insufficient_data.append(f"{country}: not in data")
            continue

        country_data = country_market[country_market['country'] == country]
        if country_data.empty:
            countries_with_insufficient_data.append(f"{country}: insufficient data")
            continue

        baseline_forecast = build_baseline_dataframe(
            country_data,
            forecast_until,
            derived_hist_cutoff
        )

        if baseline_forecast.empty:
            countries_with_insufficient_data.append(f"{country}: insufficient data")
            continue

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

        adjusted_forecast, adjustment_details = calculate_unified_adjustment(
            baseline_forecast,
            country_news_df,
            None,
            cfg,
            news_confidence_multiplier=news_confidence
        )

        adjusted_forecast, adjustment_details = apply_adjustment_window(
            adjusted_forecast,
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

        country_metadata = {
            'data_points': len(country_data),
            'has_indicators': False,
            'has_country_news': country_news_df is not None and not country_news_df.empty,
            'news_articles_count': len(country_news_df) if isinstance(country_news_df, pd.DataFrame) else 0,
            'adjustments_applied': list(adjustments_applied),
            'adjustment_details': adjustment_details,
            'fallback_used': fallback_reason is not None,
            'fallback_reason': fallback_reason
        }

        country_forecasts[country] = {
            'forecast': adjusted_forecast,
            'baseline': baseline_forecast,
            'metadata': country_metadata
        }

    if not country_forecasts:
        error_details = "; ".join(countries_with_insufficient_data)
        return {'error': f'No countries have sufficient data for forecasting. Details: {error_details}'}

    metadata = {
        'countries_forecasted': len(country_forecasts),
        'countries_requested': len(selected_countries),
        'countries_with_news': countries_with_news,
        'countries_indicators_only': countries_indicators_only,
        'countries_with_insufficient_data': countries_with_insufficient_data,
        'news_coverage': f"{len(countries_with_news)}/{len(country_forecasts)} countries",
        'forecast_years': list(range((derived_hist_cutoff or 0) + 1, int(forecast_until) + 1)),
        'adjustment_breakdown': all_adjustment_details
    }

    return {
        'country_forecasts': country_forecasts,
        'method': 'country_specific',
        'adjustments_applied': list(adjustments_applied),
        'adjustment_details': all_adjustment_details,
        'metadata': metadata
    }
