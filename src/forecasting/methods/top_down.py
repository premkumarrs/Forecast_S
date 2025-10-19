"""
Top-down forecast method.
Pipeline: Global forecast -> Distribution to countries -> Country adjustments -> Regional Aggregation
"""

import pandas as pd
import logging
from typing import Dict, Optional

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

def forecast_top_down(
    unified_data: Dict,
    config: Dict,
    analyzed_news: Optional[pd.DataFrame] = None,
    country_news: Optional[Dict[str, pd.DataFrame]] = None
) -> Dict:
    """
    Top-down: A baseline global forecast is distributed to countries,
    then adjusted for each country, and finally re-aggregated.
    """
    logger.info("Starting Top-Down forecast method")

    if config.get('forecast_mode', ForecastMode.CLASSIC.value) == ForecastMode.EXISTING_FORECAST_NEWS.value:
        return _forecast_existing_news_top_down(
            unified_data,
            config,
            analyzed_news,
            country_news
        )

    # Step 1: Generate a BASELINE global forecast (no adjustments)
    global_market = unified_data.get('market_value', {}).get('global', pd.DataFrame())
    if global_market.empty:
        return {'error': 'No global market data available for top-down forecast'}

    baseline_global_forecast = BaselineModelFactory.generate_baseline_forecast(
        global_market,
        config.get('forecast_method', '3-yr CAGR'),
        config['hist_cutoff'],
        config['forecast_until']
    )

    # Step 2: Distribute the baseline forecast to countries
    country_market = unified_data.get('market_value', {}).get('country', pd.DataFrame())
    if country_market.empty:
        logger.warning("No country market data for distribution; returning global-only baseline.")
        return {'global_forecast': baseline_global_forecast, 'method': 'top_down', 'adjustments_applied': []}

    country_shares = _calculate_country_shares(country_market, config['hist_cutoff'])
    country_indicators = unified_data.get('indicators', {}).get('country', pd.DataFrame())
    
    adjusted_country_forecasts = {}
    baseline_country_forecasts = {}  # Store baseline forecasts separately
    all_adjustment_details = {}
    adjustments_applied = set()
    fallback_countries = {}  # Track which countries use fallback

    # Step 3: Adjust each country's baseline forecast
    for country, _share in country_shares.items():
        baseline_country_forecast = baseline_global_forecast.copy()
        baseline_country_forecast['value_hat'] *= share
        
        # Store baseline for later reference
        baseline_country_forecasts[country] = baseline_country_forecast.copy()

        country_news_df = country_news.get(country) if country_news is not None else None
        if country_news_df is None:
            country_news_df = pd.DataFrame()
        news_confidence = 1.0
        fallback_reason = None

        # Apply fallback if country news is insufficient
        if country_news_df.empty or len(country_news_df) < 5:
            country_news_df, news_confidence, fallback_reason = apply_fallback_strategy(
                country,
                analyzed_news, # Global news for fallback
                config
            )
            fallback_countries[country] = fallback_reason

        country_indicators_df = country_indicators[country_indicators['country'] == country] if not country_indicators.empty else None

        # Just-in-time override from calibration (country-specific → fallback to global)
        cfg = dict(config)
        cal_scope = {}
        try:
            # Prefer country-specific calibration attached to DF
            cal_scope = getattr(country_news_df, 'attrs', {}).get('calibration', {}) or {}
            if not cal_scope and isinstance(analyzed_news, pd.DataFrame):
                cal_scope = getattr(analyzed_news, 'attrs', {}).get('calibration', {}) or {}
        except Exception:
            cal_scope = {}
        if cal_scope:
            aw = dict(cfg.get('adjustment_weights', {}))
            cfg['long_term_decay_rate'] = cal_scope.get('long_term_decay_rate', cfg.get('long_term_decay_rate', 0.60))
            cfg['adjustment_weights'] = aw

        adjusted_forecast, details = calculate_unified_adjustment(
            baseline_country_forecast,
            country_news_df,
            country_indicators_df,
            cfg,
            news_confidence_multiplier=news_confidence
        )
        
        # Add fallback info to details
        if fallback_reason:
            details['fallback_used'] = True
            details['fallback_reason'] = fallback_reason
        else:
            details['fallback_used'] = False
        
        adjusted_country_forecasts[country] = adjusted_forecast
        all_adjustment_details[country] = details
        if details.get('news_contribution', 0) != 0:
            adjustments_applied.add('news')
        if details.get('indicator_contribution', 0) != 0:
            adjustments_applied.add('indicators')

    # Step 4: Re-aggregate adjusted country forecasts
    region_forecasts = aggregate_countries_to_regions(adjusted_country_forecasts)
    final_global_forecast = region_forecasts.get('Worldwide', baseline_global_forecast)

    return {
        'global_forecast': final_global_forecast,
        'baseline_global_forecast': baseline_global_forecast,  # Add baseline global
        'region_forecasts': region_forecasts,
        'country_forecasts': adjusted_country_forecasts,
        'baseline_country_forecasts': baseline_country_forecasts,  # Add baseline countries
        'method': 'top_down',
        'adjustments_applied': list(adjustments_applied),
        'adjustment_details': all_adjustment_details,
        'metadata': {
            'countries_forecasted': len(adjusted_country_forecasts),
            'regions_aggregated': len(region_forecasts),
            'country_shares': country_shares,
            'adjustment_breakdown': all_adjustment_details,
            'fallback_countries': fallback_countries,
            'countries_with_fallback': len(fallback_countries)
        }
    }


def _forecast_existing_news_top_down(
    unified_data: Dict,
    config: Dict,
    analyzed_news: Optional[pd.DataFrame],
    country_news: Optional[Dict[str, pd.DataFrame]]
) -> Dict:
    forecast_until = config.get('forecast_until')
    if forecast_until is None:
        return {'error': 'forecast_until is required for existing forecasts mode'}

    global_market = unified_data.get('market_value', {}).get('global', pd.DataFrame())
    country_market = unified_data.get('market_value', {}).get('country', pd.DataFrame())

    if global_market is None or global_market.empty:
        return {'error': 'No global market data available for top-down forecast'}
    if country_market is None or country_market.empty:
        return {'error': 'No country market data available for top-down forecast'}

    start_year = config.get('adjustment_start_year')
    end_year = config.get('adjustment_end_year')
    derived_hist_cutoff = derive_hist_cutoff(config)

    baseline_global_forecast = build_baseline_dataframe(
        global_market,
        forecast_until,
        derived_hist_cutoff
    )


    country_shares = _calculate_country_shares(country_market, derived_hist_cutoff)

    adjusted_country_forecasts = {}
    baseline_country_forecasts = {}
    all_adjustment_details = {}
    adjustments_applied = set()
    fallback_countries = {}

    cfg = dict(config)
    cfg['hist_cutoff'] = derived_hist_cutoff
    cfg['use_indicators'] = False
    cfg['use_news'] = True

    for country, share in country_shares.items():
        country_data = country_market[country_market['country'] == country]
        if country_data.empty:
            continue

        baseline_country_forecast = build_baseline_dataframe(
            country_data,
            forecast_until,
            derived_hist_cutoff
        )

        if baseline_country_forecast.empty:
            continue

        baseline_country_forecasts[country] = baseline_country_forecast.copy()

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
            if fallback_reason:
                fallback_countries[country] = fallback_reason

        adjusted_forecast, adjustment_details = calculate_unified_adjustment(
            baseline_country_forecast,
            country_news_df,
            None,
            cfg,
            news_confidence_multiplier=news_confidence
        )

        adjusted_forecast, adjustment_details = apply_adjustment_window(
            adjusted_forecast,
            baseline_country_forecast,
            adjustment_details,
            start_year,
            end_year
        )

        if adjustment_details and adjustment_details.get('news_contribution', 0) != 0:
            adjustments_applied.add('news')

        all_adjustment_details[country] = adjustment_details
        adjusted_country_forecasts[country] = adjusted_forecast

    if not adjusted_country_forecasts:
        return {'error': 'No countries produced forecasts for existing forecasts mode'}

    region_forecasts = aggregate_countries_to_regions(adjusted_country_forecasts)
    final_global_forecast = region_forecasts.get('Worldwide')
    if final_global_forecast is None:
        final_global_forecast = baseline_global_forecast.copy()

    metadata = {
        'countries_forecasted': len(adjusted_country_forecasts),
        'regions_aggregated': len(region_forecasts),
        'country_shares': country_shares,
        'adjustment_breakdown': all_adjustment_details,
        'fallback_countries': fallback_countries,
        'countries_with_fallback': len(fallback_countries)
    }

    return {
        'global_forecast': final_global_forecast,
        'baseline_global_forecast': baseline_global_forecast,
        'region_forecasts': region_forecasts,
        'country_forecasts': adjusted_country_forecasts,
        'baseline_country_forecasts': baseline_country_forecasts,
        'method': 'top_down',
        'adjustments_applied': list(adjustments_applied),
        'adjustment_details': all_adjustment_details,
        'metadata': metadata
    }

def _calculate_country_shares(country_market: pd.DataFrame, hist_cutoff: int) -> Dict[str, float]:
    """Calculate historical market shares for each country."""
    historical_data = country_market[country_market['year'] <= hist_cutoff]
    if historical_data.empty:
        countries = country_market['country'].unique()
        return {country: 1.0 / len(countries) for country in countries} if countries.any() else {}

    country_totals = historical_data.groupby('country')['value'].sum()
    total_market = country_totals.sum()
    if total_market == 0:
        countries = country_market['country'].unique()
        return {country: 1.0 / len(countries) for country in countries} if countries.any() else {}

    return (country_totals / total_market).to_dict()
