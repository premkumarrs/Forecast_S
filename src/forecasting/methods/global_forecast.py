"""
Global-only forecast method.
Pipeline: Global data → Baseline → Adjustments → Final forecast
"""

import pandas as pd
import logging
from typing import Dict, Optional

from ..models.baseline_factory import BaselineModelFactory
from ..adjustments import calculate_unified_adjustment
from ...constants import ForecastMode
from .existing_mode_helpers import (
    build_baseline_dataframe,
    apply_adjustment_window,
    derive_hist_cutoff,
)

logger = logging.getLogger(__name__)


def _resolve_news_data(analyzed_news):
    """Normalize analyzed news payloads into a DataFrame when available."""
    news_data = None
    news_format = "none"

    if isinstance(analyzed_news, dict):
        news_type = analyzed_news.get('type')
        if news_type == 'global_news':
            news_data = analyzed_news.get('data')
            news_format = "dictionary_global_news"
        else:
            news_data = analyzed_news
            news_format = f"dictionary_{news_type or 'unknown'}"
    elif isinstance(analyzed_news, pd.DataFrame):
        news_data = analyzed_news
        news_format = "dataframe"
    elif analyzed_news is None:
        news_format = "none"
    else:
        news_data = analyzed_news
        news_format = f"unknown_{type(analyzed_news).__name__}"

    return news_data, news_format


def _forecast_existing_news_global(unified_data: Dict, config: Dict, analyzed_news: Optional[pd.DataFrame]) -> Dict:
    forecast_until = config.get('forecast_until')
    if forecast_until is None:
        return {'error': 'forecast_until is required for existing forecasts mode'}

    global_market = unified_data.get('market_value', {}).get('global', pd.DataFrame())
    if global_market is None or global_market.empty:
        return {'error': 'No global market data available'}

    start_year = config.get('adjustment_start_year')
    end_year = config.get('adjustment_end_year')
    derived_hist_cutoff = derive_hist_cutoff(config)

    baseline_forecast = build_baseline_dataframe(
        global_market,
        forecast_until,
        derived_hist_cutoff,
    )

    cfg = dict(config)
    cfg['hist_cutoff'] = derived_hist_cutoff
    cfg['use_indicators'] = False
    cfg['use_news'] = True

    news_data, news_format = _resolve_news_data(analyzed_news)
    indicators_df = None

    adjusted_forecast, adjustment_details = calculate_unified_adjustment(
        baseline_forecast,
        news_data,
        indicators_df,
        cfg
    )

    adjusted_forecast, adjustment_details = apply_adjustment_window(
        adjusted_forecast,
        baseline_forecast,
        adjustment_details,
        start_year,
        end_year
    )

    adjustments_applied = []
    if adjustment_details and adjustment_details.get('news_contribution', 0) != 0:
        adjustments_applied.append('news')

    metadata = {
        'data_points': len(global_market),
        'has_indicators': False,
        'has_news': news_data is not None and hasattr(news_data, 'empty') and not getattr(news_data, 'empty', False),
        'news_articles_count': len(news_data) if isinstance(news_data, pd.DataFrame) else 0,
        'news_format': news_format,
        'news_applied': 'news' in adjustments_applied,
        'forecast_years': list(range((derived_hist_cutoff or 0) + 1, int(forecast_until) + 1)),
        'adjustment_breakdown': {'global': adjustment_details}
    }

    return {
        'global_forecast': adjusted_forecast,
        'forecast': adjusted_forecast,
        'baseline': baseline_forecast,
        'analyzed_news': news_data if isinstance(news_data, pd.DataFrame) else pd.DataFrame(),
        'adjustment_details': {'global': adjustment_details},
        'method': 'global_only',
        'adjustments_applied': adjustments_applied,
        'metadata': metadata
    }


def forecast_global_only(
    unified_data: Dict,
    config: Dict,
    analyzed_news: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Global-only forecast without country distribution.
    
    Args:
        unified_data: Unified data structure
        config: Configuration dictionary
        analyzed_news: Global news analysis (DataFrame or dict with 'data' key)
    
    Returns:
        Dictionary with forecast results
    """
    logger.info("Starting Global-Only forecast method")
    logger.info(f"News data type received: {type(analyzed_news)}")

    forecast_mode = config.get('forecast_mode', ForecastMode.CLASSIC.value)
    if forecast_mode == ForecastMode.EXISTING_FORECAST_NEWS.value:
        return _forecast_existing_news_global(unified_data, config, analyzed_news)

    news_data, news_format = _resolve_news_data(analyzed_news)
    if news_format == "none":
        logger.info("No news data provided")
    elif news_format.startswith("unknown_"):
        logger.warning(f"Unexpected news data format: {type(analyzed_news)}")
    elif news_format.startswith("dictionary"):
        logger.info("Extracted news from dictionary format")
    elif news_format == "dataframe":
        logger.info("Using traditional DataFrame format for news")

    # Extract global data
    global_market = unified_data.get('market_value', {}).get('global', pd.DataFrame())
    global_indicators = unified_data.get('indicators', {}).get('global', pd.DataFrame())
    
    if global_market.empty:
        return {'error': 'No global market data available'}
    
    # Step 1: Generate baseline forecast
    baseline_forecast = BaselineModelFactory.generate_baseline_forecast(
        global_market,
        config.get('forecast_method', '3-yr CAGR'),
        config['hist_cutoff'],
        config['forecast_until']
    )
    
    # Step 2: Apply unified adjustment (override impact/decay via calibration if available)
    cfg = dict(config)
    cal_scope = {}
    # Extract calibration from dict payload or DataFrame attrs
    if isinstance(analyzed_news, dict):
        cal_scope = (analyzed_news.get('calibration') or {}).get('global', {})
        news_data = analyzed_news.get('data') if analyzed_news.get('type') == 'global_news' else news_data
    elif isinstance(news_data, pd.DataFrame):
        try:
            cal_scope = getattr(news_data, 'attrs', {}).get('calibration', {})
        except Exception:
            cal_scope = {}
    # Inject overrides just-in-time
    aw = dict(cfg.get('adjustment_weights', {}))
    if cal_scope:
        cfg['long_term_decay_rate'] = cal_scope.get('long_term_decay_rate', cfg.get('long_term_decay_rate', 0.60))
        cfg['adjustment_weights'] = aw

    adjusted_forecast, adjustment_details = calculate_unified_adjustment(
        baseline_forecast,
        news_data,
        global_indicators,
        cfg
    )

    # Store what was applied
    adjustments_applied = []
    if adjustment_details.get('news_contribution', 0) != 0:
        adjustments_applied.append('news')
    if adjustment_details.get('indicator_contribution', 0) != 0:
        adjustments_applied.append('indicators')

    logger.info(f"Global-Only forecast complete: {adjustments_applied}")

    return {
        'global_forecast': adjusted_forecast,
        'forecast': adjusted_forecast,  # Keep both for compatibility
        'baseline': baseline_forecast,
        'analyzed_news': news_data if isinstance(news_data, pd.DataFrame) else pd.DataFrame(),
        'adjustment_details': {'global': adjustment_details},
        'method': 'global_only',
        'adjustments_applied': adjustments_applied,
        'metadata': {
            'data_points': len(global_market),
            'has_indicators': not global_indicators.empty,
            'has_news': news_data is not None and hasattr(news_data, 'empty') and not news_data.empty,
            'news_articles_count': len(news_data) if isinstance(news_data, pd.DataFrame) else 0,
            'news_format': news_format,
            'news_applied': 'news' in adjustments_applied,
            'forecast_years': list(range(config['hist_cutoff'] + 1, config['forecast_until'] + 1)),
            'adjustment_breakdown': {'global': adjustment_details}
        }
    }
