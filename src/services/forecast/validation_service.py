"""
Validation service for forecast operations.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional
from ...forecasting.validation import validate_method_data, get_method_requirements
from ..base import BaseService

logger = logging.getLogger(__name__)


class ForecastValidationService(BaseService):
    """Service for validating forecast data and methods."""
    
    def __init__(self):
        """Initialize validation service."""
        super().__init__("ForecastValidation")
    
    def _initialize(self) -> None:
        """No special initialization needed."""
        pass
    
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
            Validation result dictionary
        """
        return validate_method_data(method, unified_data, selected_countries)
    
    def get_method_info(self, method: str) -> Dict:
        """
        Get information about method requirements.
        
        Args:
            method: Method name
            
        Returns:
            Method requirements dictionary
        """
        return get_method_requirements(method)
    
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
        from .method_router import MethodRouter
        
        available_methods = []
        
        for method in MethodRouter.get_all_methods():
            # Pass selected_countries for Country-Specific method
            countries = selected_countries if method == 'Country-Specific' else None
            validation = self.validate_method(method, unified_data, countries)
            if validation['valid']:
                available_methods.append(method)
        
        return available_methods
    
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
        report = {
            'compatible': True,
            'optimal': True,
            'missing_components': [],
            'recommendations': [],
            'news_coverage': self.calculate_news_coverage(analyzed_news, method)
        }
        
        if not analyzed_news:
            report['compatible'] = True  # News is optional
            report['optimal'] = False
            report['missing_components'].append('No news data available')
            report['recommendations'].append(
                'Consider fetching news for enhanced forecast adjustments'
            )
            return report
        
        news_type = analyzed_news.get('type')
        from .method_router import MethodRouter
        required_news = MethodRouter.get_news_routing(method)
        
        # Check compatibility based on method requirements
        if required_news == 'global' and news_type != 'global_news':
            if news_type != 'combined_news' or not analyzed_news.get('global_data'):
                report['optimal'] = False
                report['missing_components'].append('Global news')
                
        elif required_news == 'country' and news_type not in ['country_news', 'combined_news']:
            report['optimal'] = False
            report['missing_components'].append('Country news')
            
        elif required_news == 'combined':
            if news_type != 'combined_news':
                report['optimal'] = False
                if news_type == 'global_news':
                    report['missing_components'].append('Country news')
                elif news_type == 'country_news':
                    report['missing_components'].append('Global news')
        
        # Add specific recommendations
        if not report['optimal']:
            if method == 'Top-Down' and 'Country news' in report['missing_components']:
                report['recommendations'].append(
                    'Top-Down method will use uniform distribution without country news'
                )
            elif method in ['Bottom-Up', 'Country-Specific'] and news_type == 'global_news':
                report['recommendations'].append(
                    f'{method} requires country-specific news for optimal results'
                )
        
        return report
    
    def calculate_news_coverage(
        self,
        analyzed_news: Optional[Dict],
        method: str
    ) -> Dict:
        """
        Calculate statistics about news coverage.
        
        Args:
            analyzed_news: Analyzed news data
            method: Forecasting method name
            
        Returns:
            News coverage statistics
        """
        coverage = {
            'has_global_news': False,
            'has_country_news': False,
            'global_article_count': 0,
            'country_article_counts': {},
            'total_articles': 0,
            'warnings': []
        }
        
        if not analyzed_news or not isinstance(analyzed_news, dict):
            coverage['warnings'].append('No news data available')
            return coverage
        
        news_type = analyzed_news.get('type')
        
        # Process based on news type
        if news_type == 'combined_news':
            self._process_combined_news(analyzed_news, coverage)
        elif news_type == 'global_news':
            self._process_global_news(analyzed_news, coverage)
        elif news_type == 'country_news':
            self._process_country_news(analyzed_news, coverage)
        
        # Add method-specific warnings
        self._add_method_warnings(method, coverage)
        
        return coverage
    
    def _process_combined_news(self, analyzed_news: Dict, coverage: Dict) -> None:
        """Process combined news format."""
        global_data = analyzed_news.get('global_data')
        if global_data is not None and hasattr(global_data, '__len__'):
            if len(global_data) > 0:
                coverage['has_global_news'] = True
                coverage['global_article_count'] = len(global_data)
                coverage['total_articles'] += len(global_data)
        
        country_data = analyzed_news.get('country_data', {})
        if country_data:
            coverage['has_country_news'] = True
            for country, df in country_data.items():
                if df is not None and hasattr(df, '__len__'):
                    if len(df) > 0:
                        count = len(df)
                        coverage['country_article_counts'][country] = count
                        coverage['total_articles'] += count
    
    def _process_global_news(self, analyzed_news: Dict, coverage: Dict) -> None:
        """Process global news format."""
        data = analyzed_news.get('data')
        if data is not None and hasattr(data, '__len__'):
            if len(data) > 0:
                coverage['has_global_news'] = True
                coverage['global_article_count'] = len(data)
                coverage['total_articles'] = len(data)
    
    def _process_country_news(self, analyzed_news: Dict, coverage: Dict) -> None:
        """Process country news format."""
        data = analyzed_news.get('data', {})
        if data:
            coverage['has_country_news'] = True
            for country, df in data.items():
                if df is not None and hasattr(df, '__len__'):
                    if len(df) > 0:
                        count = len(df)
                        coverage['country_article_counts'][country] = count
                        coverage['total_articles'] += count
    
    def _add_method_warnings(self, method: str, coverage: Dict) -> None:
        """Add method-specific warnings to coverage."""
        if method == 'Top-Down':
            if not coverage['has_global_news']:
                coverage['warnings'].append(
                    'Top-Down method missing global news for optimal global forecast'
                )
            if not coverage['has_country_news']:
                coverage['warnings'].append(
                    'Top-Down method missing country news for distribution adjustments'
                )
        elif method in ['Bottom-Up', 'Country-Specific']:
            if not coverage['has_country_news']:
                coverage['warnings'].append(
                    f'{method} method missing country news for individual country analysis'
                )
        elif method == 'Global Only':
            if not coverage['has_global_news']:
                coverage['warnings'].append(
                    'Global Only method missing global news for adjustments'
                )