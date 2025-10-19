"""
Main data extraction service for database operations.
"""

import pandas as pd
import logging
from typing import Dict, List, Tuple
from datetime import datetime

from .processor import DataProcessor
from .validator import DataValidator
from .mock_data import MockDataGenerator
from ...constants import ForecastMode

logger = logging.getLogger(__name__)


class DataExtractionService:
    """Handles data extraction operations."""
    
    def __init__(self):
        self.processor = DataProcessor()
        self.validator = DataValidator()
        self.mock_generator = MockDataGenerator()
    
    def validate_extraction_params(self, market_kpi: str, indicator_kpis: List[str], 
                                  hist_cutoff: int, forecast_until: int) -> Dict:
        """Validate extraction parameters."""
        return self.validator.validate_params(market_kpi, indicator_kpis, hist_cutoff, forecast_until)
    
    def extract_unified_data(self, market_kpi: str, indicator_kpis: List[str], 
                            hist_cutoff: int, forecast_until: int) -> Dict:
        """Extract unified data structure."""
        try:
            unified_data = {}
            
            # Extract market data
            include_forecast_years = False
            min_year = None

            try:
                import streamlit as st

                mode_value = st.session_state.get('forecast_mode', ForecastMode.CLASSIC.value)
                include_forecast_years = mode_value == ForecastMode.EXISTING_FORECAST_NEWS.value
            except Exception:
                include_forecast_years = False

            market_result = self.processor.extract_market_data(
                market_kpi,
                hist_cutoff,
                forecast_until,
                include_forecast_years=include_forecast_years,
                min_year=min_year
            )
            if 'error' in market_result:
                return market_result
            
            unified_data['market_value'] = market_result
            
            # Extract indicator data if provided
            if indicator_kpis:
                indicator_result = self.processor.extract_indicator_data(
                    indicator_kpis, hist_cutoff, forecast_until
                )
                if 'error' in indicator_result:
                    # Don't fail completely, just log warning
                    unified_data['indicators'] = {'global': pd.DataFrame(), 'country': pd.DataFrame()}
                    unified_data['extraction_warnings'] = [
                        f"Indicator extraction failed: {indicator_result['error']}"
                    ]
                else:
                    unified_data['indicators'] = indicator_result
            else:
                unified_data['indicators'] = {'global': pd.DataFrame(), 'country': pd.DataFrame()}
            
            # Add extraction metadata
            unified_data['extraction_params'] = {
                'market_kpi': market_kpi,
                'indicator_kpis': indicator_kpis,
                'hist_cutoff': hist_cutoff,
                'forecast_until': forecast_until,
                'extraction_timestamp': datetime.now()
            }
            
            return unified_data
            
        except Exception as e:
            return {'error': f'Data extraction failed: {str(e)}'}
    
    def get_extraction_summary(self, unified_data: Dict) -> Dict:
        """Generate summary of extracted data."""
        return self.processor.get_data_summary(unified_data)
    
    def create_mock_data(self, market_kpi: str, indicator_kpis: List[str], 
                        hist_cutoff: int, forecast_until: int) -> Dict:
        """Create mock data for demo purposes."""
        return self.mock_generator.create_mock_data(
            market_kpi, indicator_kpis, hist_cutoff, forecast_until
        )
    
    def get_available_countries(self, unified_data: Dict) -> List[str]:
        """Get list of available countries from unified data."""
        return self.processor.get_available_countries(unified_data)
    
    def has_database_connection(self) -> bool:
        """Check if database connection is available."""
        return self.processor.has_database_connection()
