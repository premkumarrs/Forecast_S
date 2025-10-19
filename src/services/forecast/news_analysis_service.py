"""
News analysis service for forecast adjustments.
"""

import os
import pandas as pd
import logging
from typing import Dict, List, Optional
from ...llm.analyst import SimpleLLMAnalyst as LLM_Analyst
from ...llm.calibrators.impact_decay_calibrator import (
    DecayCalibrator,
    recency_weighted_avg_pct,
)
from ...news import fetch_max_gdelt_articles, fetch_multi_country_headlines
from ...data import extract_country_iso3_mapping
from ..base import BaseService

logger = logging.getLogger(__name__)


class NewsAnalysisService(BaseService):
    """Service for fetching and analyzing news data."""
    
    def __init__(self, market_name: str):
        """
        Initialize news analysis service.
        
        Args:
            market_name: Name of the market for analysis
        """
        super().__init__("NewsAnalysis")
        self.market_name = market_name
        self.llm_analyst = None
        self.calibrator = DecayCalibrator()
        self.article_threshold = int(os.getenv('NEWS_ARTICLE_THRESHOLD', '10'))
    
    def _initialize(self) -> None:
        """Initialize LLM analyst."""
        if not self.llm_analyst:
            self.llm_analyst = LLM_Analyst(self.market_name)
    
    def fetch_global_news(
        self,
        topics: List[str],
        progress_callback=None,
        method: str = None,
        forecast_years: int = None,
        config: Optional[Dict] = None,
    ) -> Dict:
        """
        Fetch and analyze global news.
        
        Args:
            topics: List of topics to search
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary with analyzed global news
        """
        if not topics:
            return {'error': 'No topics configured for news analysis'}
        
        # Progress wrapper for global fetch
        def global_fetch_progress(completed, total, batches_count, country_fips):
            if progress_callback:
                progress_callback(
                    f"Fetching global articles: {completed}/{total} batches, "
                    f"{batches_count} articles found"
                )
        
        # Fetch global headlines
        global_headlines, _ = fetch_max_gdelt_articles(
            topics,
            progress_callback=global_fetch_progress
        )
        
        if global_headlines.empty:
            return {'error': 'No news articles found for the specified topics'}
        
        # Initialize LLM if needed
        self.initialize()
        
        # Progress wrapper for LLM analysis
        def global_llm_progress(completed, total, status):
            if progress_callback:
                progress_callback(
                    f"Analyzing global headlines: {completed}/{total} ({status})"
                )
        
        # Analyze headlines
        analyzed_news = self.llm_analyst.analyze_headlines(
            global_headlines.to_dict('records'),
            progress_callback=global_llm_progress
        )
        
        logger.info(f"Global news analysis complete: {len(analyzed_news)} results")
        
        # Build calibration (global scope) using MA-based calculation
        if progress_callback:
            progress_callback("Calculating category-based moving average impact...")
        
        # Calculate using MA system with categories
        from src.forecasting.adjustments.news_adjustment import NewsAdjustment
        news_adj = NewsAdjustment(weight=1.0)
        
        # Get categories from passed config or session state
        categories = []
        if config and 'categories' in config:
            categories = config.get('categories', [])
        else:
            # Fallback to session state if config not provided
            try:
                import streamlit as st
                session_config = st.session_state.get('config', {})
                categories = session_config.get('categories', [])
            except:
                categories = []
        
        # Calculate MA-based impact with recency weighting
        if categories:
            config_for_ma = {'categories': categories}
            g_avg = news_adj.calculate(analyzed_news, config_for_ma) * 100  # Convert to percentage
        else:
            # Fallback to recency-weighted average if no categories
            try:
                g_avg = recency_weighted_avg_pct(analyzed_news, half_life_days=int(os.getenv('NEWS_HALF_LIFE_DAYS', '90')))
            except Exception:
                g_avg = 0.0
        
        # Get category breakdown if available
        category_breakdown = getattr(news_adj, 'category_breakdown', None) if categories else None
        
        # Forecast years unknown here; use default inside calibrator
        cal_global = self.calibrator.calibrate_global(self.market_name, g_avg, forecast_years=forecast_years, method=method)
        logger.info(f"Calibration (global): avg={g_avg:.3f} (MA-based with recency weighting)")
        calibration = {
            'global': {**cal_global, 'news_avg_pct': float(g_avg), 'category_breakdown': category_breakdown} if category_breakdown else {**cal_global, 'news_avg_pct': float(g_avg)},
            'countries': {}
        }
        # Attach calibration to DataFrame attrs for downstream access
        try:
            analyzed_news.attrs['calibration'] = calibration['global']
        except Exception:
            pass

        return {
            'type': 'global_news',
            'data': analyzed_news,
            'count': len(analyzed_news),
            'calibration': calibration
        }
    
    def fetch_country_news(
        self,
        countries: List[str],
        topics: List[str],
        unified_data: Dict,
        progress_callback=None,
        method: str = None,
        forecast_years: int = None,
        config: Optional[Dict] = None,
    ) -> Dict:
        """
        Fetch and analyze country-specific news.
        
        Args:
            countries: List of countries
            topics: List of topics to search
            unified_data: Unified data containing ISO3 mappings
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary with analyzed country news
        """
        # Get ISO3 mapping
        country_iso3_full = extract_country_iso3_mapping(unified_data)
        logger.info(f"ISO3 mapping for {len(countries)} countries")
        
        # Create mapping for selected countries
        selected_country_iso3 = self._build_country_iso3_mapping(
            countries,
            country_iso3_full
        )
        
        # Progress wrapper for country fetching
        def fetch_progress(completed, total, country_name, article_count):
            if progress_callback:
                progress_callback(
                    f"Fetched {article_count} articles from {country_name} "
                    f"({completed}/{total} countries)"
                )
        
        # Fetch headlines for all countries
        country_headlines = fetch_multi_country_headlines(
            selected_country_iso3,
            topics,
            progress_callback=fetch_progress
        )
        
        # Analyze headlines per country
        country_news = self._analyze_country_headlines(
            country_headlines,
            progress_callback
        )
        
        # Calculate statistics
        total_articles = sum(
            len(news) for news in country_news.values()
        )
        countries_below_threshold = {
            country: len(headlines)
            for country, headlines in country_headlines.items()
            if len(headlines) < self.article_threshold
        }
        
        logger.info(
            f"Country news analysis complete: {total_articles} total results "
            f"across {len(country_news)} countries"
        )
        
        # Build per-country calibration using MA system
        calibration = {
            'global': {},  # not available in this call
            'countries': {}
        }
        if progress_callback:
            progress_callback("Calculating category-based moving average impact for countries...")
        
        # Get categories from passed config or session state
        categories = []
        if config and 'categories' in config:
            categories = config.get('categories', [])
        else:
            # Fallback to session state if config not provided
            try:
                import streamlit as st
                session_config = st.session_state.get('config', {})
                categories = session_config.get('categories', [])
            except:
                categories = []
        
        for cname, cdf in country_news.items():
            # Calculate using MA system with categories
            from src.forecasting.adjustments.news_adjustment import NewsAdjustment
            news_adj = NewsAdjustment(weight=1.0)
            
            if categories:
                config_for_ma = {'categories': categories}
                c_avg = news_adj.calculate(cdf, config_for_ma) * 100  # Convert to percentage
                category_breakdown = getattr(news_adj, 'category_breakdown', None)
            else:
                # Fallback to recency-weighted average if no categories
                try:
                    c_avg = recency_weighted_avg_pct(cdf, half_life_days=int(os.getenv('NEWS_HALF_LIFE_DAYS', '90')))
                    category_breakdown = None
                except Exception:
                    c_avg = 0.0
                    category_breakdown = None
            
            cal = self.calibrator.calibrate_country(self.market_name, cname, c_avg, forecast_years=forecast_years, method=method)
            logger.info(f"Calibration (country={cname}): avg={c_avg:.3f} → decay={cal.get('long_term_decay_rate')} (MA-based)")
            calibration['countries'][cname] = {**cal, 'news_avg_pct': float(c_avg), 'category_breakdown': category_breakdown} if category_breakdown else {**cal, 'news_avg_pct': float(c_avg)}
            # Attach per-DF attrs for downstream access
            try:
                cdf.attrs['calibration'] = calibration['countries'][cname]
            except Exception:
                pass

        return {
            'type': 'country_news',
            'data': country_news,
            'count': total_articles,
            'countries': list(country_news.keys()),
            'countries_below_threshold': countries_below_threshold,
            'threshold': self.article_threshold,
            'indicators_only_countries': list(countries_below_threshold.keys()),
            'calibration': calibration
        }
    
    def fetch_combined_news(
        self,
        topics: List[str],
        unified_data: Dict,
        progress_callback=None,
        method: str = None,
        forecast_years: int = None,
        config: Optional[Dict] = None,
    ) -> Dict:
        """
        Fetch both global and country news (for Top-Down method).
        
        Args:
            topics: List of topics to search
            unified_data: Unified data containing country information
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary with both global and country news
        """
        # Phase 1: Fetch global news
        if progress_callback:
            progress_callback("Phase 1/2: Fetching global news...")
        
        global_result = self.fetch_global_news(topics, progress_callback, method=method, forecast_years=forecast_years, config=config)
        if 'error' in global_result:
            return global_result
        
        # Phase 2: Fetch country news
        if progress_callback:
            progress_callback("Phase 2/2: Fetching country news...")
        
        # Extract countries from market data
        country_market = unified_data.get('market_value', {}).get('country', pd.DataFrame())
        if country_market.empty:
            logger.warning("No country market data available for combined news")
            return {
                'type': 'combined_news',
                'global_data': global_result.get('data'),
                'country_data': {},
                'global_count': global_result.get('count', 0),
                'country_count': 0
            }
        
        # Get unique countries
        countries = [
            c for c in country_market['country'].unique().tolist()
            if c and isinstance(c, str) and c.strip()
        ]
        
        country_result = self.fetch_country_news(
            countries,
            topics,
            unified_data,
            progress_callback,
            method=method,
            forecast_years=forecast_years,
            config=config
        )
        # Merge calibrations
        cal = {
            'global': (global_result.get('calibration') or {}).get('global', {}),
            'countries': (country_result.get('calibration') or {}).get('countries', {})
        }
        
        return {
            'type': 'combined_news',
            'global_data': global_result.get('data'),
            'country_data': country_result.get('data', {}),
            'global_count': global_result.get('count', 0),
            'country_count': country_result.get('count', 0),
            'countries_below_threshold': country_result.get('countries_below_threshold', {}),
            'calibration': cal
        }
    
    def _build_country_iso3_mapping(
        self,
        countries: List[str],
        country_iso3_full: Dict[str, str]
    ) -> Dict[str, str]:
        """Build ISO3 mapping for selected countries."""
        selected_country_iso3 = {}
        
        for country in countries:
            if country in country_iso3_full:
                selected_country_iso3[country] = country_iso3_full[country]
                logger.info(f"  ✓ {country} → {country_iso3_full[country]}")
            else:
                # Fallback: use first 3 letters if ISO3 not found
                fallback_iso3 = (country[:3] if len(country) >= 3 else country).upper()
                selected_country_iso3[country] = fallback_iso3
                logger.warning(
                    f"  ✗ {country} → No ISO3 found, using fallback: {fallback_iso3}"
                )
        
        return selected_country_iso3
    
    def _analyze_country_headlines(
        self,
        country_headlines: Dict[str, pd.DataFrame],
        progress_callback=None
    ) -> Dict[str, pd.DataFrame]:
        """Analyze headlines for each country."""
        country_news = {}
        
        # Check if any countries have headlines to analyze
        llm_needed = any(
            not headlines.empty
            for headlines in country_headlines.values()
        )
        
        # Initialize LLM if needed
        if llm_needed:
            self.initialize()
        
        # Analyze each country's headlines
        for country, headlines_df in country_headlines.items():
            if not headlines_df.empty:
                article_count = len(headlines_df)
                logger.info(
                    f"Starting LLM analysis for {country} "
                    f"({article_count} articles)"
                )
                
                # Progress wrapper for LLM
                def llm_progress(completed, total, status):
                    if progress_callback:
                        progress_callback(
                            f"Analyzing {country}: {completed}/{total} "
                            f"headlines ({status})"
                        )
                
                # Always analyze if we have headlines
                analyzed = self.llm_analyst.analyze_headlines(
                    headlines_df.to_dict('records'),
                    country=country,
                    progress_callback=llm_progress
                )
                
                country_news[country] = analyzed
                
                # Log based on article count
                if article_count < self.article_threshold:
                    logger.info(
                        f"  ⚠️ {country}: Analysis complete - {len(analyzed)} results "
                        f"(below threshold of {self.article_threshold})"
                    )
                else:
                    logger.info(
                        f"  ✓ {country}: Analysis complete - {len(analyzed)} results"
                    )
            else:
                logger.warning(
                    f"  ✗ {country}: No articles found → INDICATORS ONLY"
                )
        
        return country_news
