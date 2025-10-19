"""
Main forecast service orchestrator - refactored version.
"""

import os
import pandas as pd
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv

from .method_router import MethodRouter
from .news_analysis_service import NewsAnalysisService
from .validation_service import ForecastValidationService
from ..base import BaseService
from ...constants import ForecastMode

logger = logging.getLogger(__name__)
load_dotenv('config/.env')


class ForecastService(BaseService):
    """Orchestrates forecasting operations with modular components."""
    
    def __init__(self):
        """Initialize forecast service."""
        super().__init__("Forecast")
        self.method_router = MethodRouter()
        self.validation_service = ForecastValidationService()
        self.news_service = None
        self.market_name = None
    
    def _initialize(self) -> None:
        """Initialize sub-services."""
        if self.market_name and not self.news_service:
            self.news_service = NewsAnalysisService(self.market_name)
    
    def set_market(self, market_name: str) -> None:
        """
        Set the market for analysis.
        
        Args:
            market_name: Name of the market
        """
        self.market_name = market_name
        if self.news_service:
            self.news_service.market_name = market_name
    
    def validate_method(
        self,
        method: str,
        unified_data: Dict,
        selected_countries: Optional[List[str]] = None
    ) -> Dict:
        """
        Validate if method can be executed with available data.
        
        Args:
            method: Forecasting method name
            unified_data: Unified data dictionary
            selected_countries: Optional list of selected countries
            
        Returns:
            Validation result
        """
        return self.validation_service.validate_method(
            method, unified_data, selected_countries
        )
    
    def get_method_info(self, method: str) -> Dict:
        """
        Get information about method requirements.
        
        Args:
            method: Method name
            
        Returns:
            Method requirements
        """
        return self.method_router.get_method_requirements(method)
    
    def get_supported_baseline_methods(self) -> List[str]:
        """Get list of supported baseline forecast methods."""
        return ["3-yr CAGR", "Damped ETS", "Logistic Growth"]
    
    def get_available_methods(
        self,
        unified_data: Dict,
        selected_countries: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get list of methods that can be executed with current data.
        
        Args:
            unified_data: Unified data dictionary
            selected_countries: Optional list of selected countries
            
        Returns:
            List of available method names
        """
        return self.validation_service.get_available_methods(
            unified_data, selected_countries
        )
    
    def analyze_news(
        self,
        method: str,
        market_name: str,
        topics: List[str],
        unified_data: Dict,
        countries: Optional[List[str]] = None,
        progress_callback=None,
        config: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze news based on the selected forecasting method.
        
        Args:
            method: Forecasting method name
            market_name: Market name for analysis
            topics: List of topics to search
            unified_data: Unified data dictionary
            countries: Optional list of countries
            progress_callback: Optional progress callback
            
        Returns:
            Analyzed news data
        """
        logger.info(f"analyze_news called with method='{method}', countries={countries}")
        
        if not topics:
            return {'error': 'No topics configured for news analysis'}
        
        # Set market if different
        if market_name != self.market_name:
            self.set_market(market_name)
        
        # Initialize news service if needed
        if not self.news_service:
            self.news_service = NewsAnalysisService(market_name)
        
        try:
            # Determine news routing strategy
            news_routing = self.method_router.get_news_routing(method)
            # Derive forecast horizon years if present in unified_data
            extraction = unified_data.get('extraction_params', {}) if isinstance(unified_data, dict) else {}
            forecast_years = None
            try:
                hist_cut = extraction.get('hist_cutoff') or extraction.get('hist_cutoff_year')
                fc_until = extraction.get('forecast_until') or extraction.get('forecast_until_year')
                if hist_cut is not None and fc_until is not None:
                    forecast_years = int(fc_until) - int(hist_cut)
            except Exception:
                forecast_years = None
            
            if news_routing == 'global':
                logger.info("Fetching global news only")
                return self.news_service.fetch_global_news(topics, progress_callback, method=method, forecast_years=forecast_years, config=config)
                
            elif news_routing == 'combined':
                logger.info("Fetching both global and country news")
                return self.news_service.fetch_combined_news(
                    topics, unified_data, progress_callback, method=method, forecast_years=forecast_years, config=config
                )
                
            elif news_routing == 'country':
                logger.info("Fetching country-specific news")
                
                # Determine countries based on method
                if method == 'Bottom-Up':
                    # Extract all countries from market data
                    country_market = unified_data.get('market_value', {}).get(
                        'country', pd.DataFrame()
                    )
                    if country_market.empty:
                        return {
                            'error': 'No country market data available for bottom-up news analysis'
                        }
                    countries = [
                        c for c in country_market['country'].unique().tolist()
                        if c and isinstance(c, str) and c.strip()
                    ]
                    logger.info(
                        f"Auto-detected {len(countries)} countries for Bottom-Up: {countries}"
                    )
                elif not countries:
                    return {
                        'error': 'No countries selected for country-specific news analysis'
                    }
                
                return self.news_service.fetch_country_news(
                    countries, topics, unified_data, progress_callback, method=method, forecast_years=forecast_years, config=config
                )
                
            else:
                return {'error': f'Unknown news routing strategy: {news_routing}'}
                
        except Exception as e:
            logger.error(f"News analysis failed for method '{method}': {str(e)}")
            return {'error': f'News analysis failed: {str(e)}'}
    
    def execute_forecast(
        self,
        method: str,
        unified_data: Dict,
        config: Dict,
        analyzed_news: Optional[Dict] = None,
        selected_countries: Optional[List[str]] = None,
        progress_callback=None
    ) -> Dict:
        """
        Execute a forecasting method.
        
        Args:
            method: Forecasting method name
            unified_data: Unified data dictionary
            config: Configuration parameters
            analyzed_news: Optional analyzed news data
            selected_countries: Optional list of selected countries
            progress_callback: Optional progress callback
            
        Returns:
            Forecast results
        """
        # Get the method function
        method_func = self.method_router.get_method_function(method)
        if not method_func:
            return {'error': f'Unknown forecasting method: {method}'}
        
        # Prepare kwargs based on method
        kwargs = self._prepare_method_kwargs(
            method, unified_data, config, analyzed_news, selected_countries
        )

        mode = config.get('forecast_mode', ForecastMode.CLASSIC.value)
        normalized_method = self.method_router.normalize_method_name(method)
        if mode == ForecastMode.EXISTING_FORECAST_NEWS.value and normalized_method == 'Top-Down':
            return {'error': 'Top-Down is unavailable in Existing Forecast + News Adjustment mode'}

        try:
            # Execute the forecast
            logger.info(f"Executing {method} forecast")
            result = method_func(**kwargs)
            
            # Add metadata - don't override method if already set
            if 'method' not in result:
                # Normalize method name for chart display compatibility
                normalized_method = method.lower().replace(' ', '_').replace('-', '_')
                result['method'] = normalized_method
            result['news_coverage'] = self.validation_service.calculate_news_coverage(
                analyzed_news, method
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Forecast execution failed for {method}: {str(e)}")
            return {'error': f'Forecast execution failed: {str(e)}'}
    
    def _prepare_method_kwargs(
        self,
        method: str,
        unified_data: Dict,
        config: Dict,
        analyzed_news: Optional[Dict],
        selected_countries: Optional[List[str]]
    ) -> Dict:
        """Prepare kwargs for method execution."""
        # Base kwargs
        kwargs = {
            'unified_data': unified_data,
            'config': config
        }
        
        # Add news data based on method
        news_kwargs = self._prepare_news_kwargs(method, analyzed_news)
        kwargs.update(news_kwargs)
        
        # Add country selection for Country-Specific method
        if method == 'Country-Specific' and selected_countries:
            kwargs['selected_countries'] = selected_countries
        
        return kwargs
    
    def _prepare_news_kwargs(
        self,
        method: str,
        analyzed_news: Optional[Dict]
    ) -> Dict:
        """Prepare news data kwargs based on method requirements."""
        kwargs = {}
        
        if not analyzed_news:
            logger.info(f"No news data available for {method}")
            return kwargs
        
        news_type = analyzed_news.get('type') if isinstance(analyzed_news, dict) else None
        logger.info(f"Preparing news data for {method}: type={news_type}")
        
        if method == 'Top-Down':
            kwargs = self._prepare_topdown_news(analyzed_news, news_type)
        elif method in ['Bottom-Up', 'Country-Specific']:
            kwargs = self._prepare_bottomup_news(analyzed_news, news_type)
        elif method == 'Global Only':
            kwargs = self._prepare_global_news(analyzed_news, news_type)
        
        return kwargs
    
    def _prepare_topdown_news(self, analyzed_news: Dict, news_type: str) -> Dict:
        """Prepare news kwargs for Top-Down method."""
        kwargs = {}
        
        if news_type == 'combined_news':
            global_data = analyzed_news.get('global_data')
            kwargs['analyzed_news'] = global_data if global_data is not None else pd.DataFrame()
            kwargs['country_news'] = analyzed_news.get('country_data', {})
            logger.info("Unpacking combined_news for Top-Down")
        elif news_type == 'global_news':
            global_data = analyzed_news.get('data')
            kwargs['analyzed_news'] = global_data if global_data is not None else pd.DataFrame()
            kwargs['country_news'] = {}
            logger.warning("Top-Down has only global news - country adjustments will be limited")
        elif news_type == 'country_news':
            kwargs['analyzed_news'] = pd.DataFrame()
            kwargs['country_news'] = analyzed_news.get('data', {})
            logger.warning("Top-Down has only country news - global forecast will use baseline only")
        
        return kwargs
    
    def _prepare_bottomup_news(self, analyzed_news: Dict, news_type: str) -> Dict:
        """Prepare news kwargs for Bottom-Up/Country-Specific methods."""
        kwargs = {
            'analyzed_news': None  # These methods use global news only for fallback
        }
        
        if news_type == 'country_news':
            kwargs['country_news'] = analyzed_news.get('data', {})
        elif news_type == 'combined_news':
            kwargs['country_news'] = analyzed_news.get('country_data', {})
        else:
            kwargs['country_news'] = {}
        
        logger.info(f"Passing country news: {len(kwargs['country_news'])} countries")
        
        return kwargs
    
    def _prepare_global_news(self, analyzed_news: Dict, news_type: str) -> Dict:
        """Prepare news kwargs for Global Only method."""
        global_data = None
        
        if news_type == 'global_news':
            global_data = analyzed_news.get('data')
        elif news_type == 'combined_news':
            global_data = analyzed_news.get('global_data')
        
        kwargs = {
            'analyzed_news': global_data if global_data is not None else pd.DataFrame()
        }
        logger.info(f"Passing global news: {len(kwargs['analyzed_news'])} articles")
        
        return kwargs
    
    def check_news_compatibility(
        self,
        method: str,
        analyzed_news: Optional[Dict],
        unified_data: Dict
    ) -> Dict:
        """
        Check if available news data is compatible with selected method.
        
        Args:
            method: Forecasting method name
            analyzed_news: Analyzed news data
            unified_data: Unified data dictionary
            
        Returns:
            Compatibility report
        """
        return self.validation_service.check_news_compatibility(
            method, analyzed_news, unified_data
        )
    
    def run_forecast(
        self,
        method: str,
        unified_data: Dict,
        config: Dict,
        analyzed_news: Optional[Dict] = None,
        selected_countries: Optional[List[str]] = None,
        progress_callback=None
    ) -> Dict:
        """
        Run a complete forecast (backward compatibility wrapper).
        
        This method wraps execute_forecast for backward compatibility.
        
        Args:
            method: Forecasting method name
            unified_data: Unified data dictionary
            config: Configuration parameters
            analyzed_news: Optional analyzed news data
            selected_countries: Optional list of selected countries
            progress_callback: Optional progress callback
            
        Returns:
            Forecast results
        """
        return self.execute_forecast(
            method=method,
            unified_data=unified_data,
            config=config,
            analyzed_news=analyzed_news,
            selected_countries=selected_countries,
            progress_callback=progress_callback
        )
