"""
Data processor for extracting and processing database data.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional
from datetime import datetime

from ...database import query, get_engine

logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes data extraction from database."""
    
    def __init__(self):
        self.excluded_regions = self._load_excluded_regions()
    
    def _load_excluded_regions(self) -> List[str]:
        """Load excluded regions from config."""
        import json
        import os
        
        config_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'config', 'exclusions.json'
        )
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get('excluded_regions', [])
        except FileNotFoundError:
            return ['European Union', 'World', 'OECD', 'G7', 'G20']
    
    def extract_market_data(
        self,
        market_kpi: str,
        hist_cutoff: int,
        forecast_until: int,
        include_forecast_years: bool = False,
        min_year: Optional[int] = None
    ) -> Dict:
        """Extract market data (global and country)."""
        try:
            # Build the query with proper parameter binding
            query_template = """
            SELECT
              kv.idGeo, kv.valueTime, kv.value,
              k.kpiKey, k.name AS kpi_name,
              g.name AS country, g.isoCode3L AS iso3
            FROM smi_contentDev.kpisValues kv
            LEFT JOIN smi_contentDev.kpis k ON kv.idKpi = k.id
            LEFT JOIN smi_contentDev.geos g ON kv.idGeo = g.id
            WHERE k.kpiKey = '{}'
            ORDER BY country, valueTime;
            """.format(market_kpi)
            
            df = query(query_template)
            
            if df.empty:
                return {'error': f'No data found for market KPI: {market_kpi}'}
            
            # Process data
            df['year'] = df['valueTime'].astype(str).str[:4].astype('int')

            if include_forecast_years:
                df_filtered = df[df['year'] <= forecast_until]
                if min_year is not None:
                    df_filtered = df_filtered[df_filtered['year'] >= min_year]
            else:
                df_filtered = df[df['year'] <= hist_cutoff]  # Only historical for market data
            
            # Remove excluded regions
            countries_before = len(df_filtered[df_filtered['idGeo'] != 100]['country'].unique())
            df_filtered = df_filtered[~df_filtered['country'].isin(self.excluded_regions)]
            countries_after = len(df_filtered[df_filtered['idGeo'] != 100]['country'].unique())
            
            # Handle duplicates
            df_dedup = df_filtered.sort_values(['country', 'year', 'valueTime']).groupby(
                ['country', 'year']
            ).last().reset_index()
            
            # Split global and country
            global_data = df_dedup[df_dedup['idGeo'] == 100][['year', 'value']].copy()
            # Keep idGeo so export can include it without re-querying
            country_data = df_dedup[df_dedup['idGeo'] != 100][
                ['idGeo', 'country', 'iso3', 'year', 'value']
            ].copy()
            
            return {
                'global': global_data,
                'country': country_data,
                'metadata': {
                    'kpi_key': market_kpi,
                    'kpi_name': df['kpi_name'].iloc[0] if not df.empty else market_kpi,
                    'total_records': len(global_data) + len(country_data),
                    'countries_count': len(country_data['country'].unique()) if not country_data.empty else 0,
                    'countries_excluded': countries_before - countries_after
                }
            }
            
        except Exception as e:
            return {'error': f'Market data extraction failed: {str(e)}'}
    
    def extract_indicator_data(self, indicator_kpis: List[str], hist_cutoff: int, 
                              forecast_until: int) -> Dict:
        """Extract indicator data (global and country)."""
        try:
            # Build the query with proper parameter binding
            placeholders = ', '.join([f"'{kpi}'" for kpi in indicator_kpis])
            
            query_template = f"""
            SELECT
              kv.idGeo,
              kv.valueTime,
              k.kpiKey AS indicator_key,
              k.name AS indicator_name,
              kv.value,
              g.name AS country,
              g.isoCode3L AS iso3
            FROM smi_contentDev.kpisValues kv
            LEFT JOIN smi_contentDev.kpis k ON kv.idKpi = k.id
            LEFT JOIN smi_contentDev.geos g ON kv.idGeo = g.id
            WHERE k.kpiKey IN ({placeholders})
            ORDER BY indicator_key, country, valueTime DESC
            """
            
            df = query(query_template)
            
            if df.empty:
                return {'error': f'No indicator data found for KPIs: {indicator_kpis}'}
            
            # Process data
            df['year'] = df['valueTime'].astype(str).str[:4].astype('int')
            df_filtered = df[(df['year'] > 2015) & (df['year'] <= forecast_until)]  # Include forecast years
            
            # Remove excluded regions
            df_filtered = df_filtered[~df_filtered['country'].isin(self.excluded_regions)]
            
            # Handle duplicates
            df_dedup = df_filtered.sort_values(
                ['indicator_key', 'country', 'year', 'valueTime']
            ).groupby(['indicator_key', 'country', 'year']).last().reset_index()
            
            # Split global and country
            global_indicators = df_dedup[df_dedup['idGeo'] == 100][
                ['year', 'indicator_key', 'indicator_name', 'value']
            ].copy()
            country_indicators = df_dedup[df_dedup['idGeo'] != 100][
                ['country', 'iso3', 'year', 'indicator_key', 'indicator_name', 'value']
            ].copy()
            
            return {
                'global': global_indicators,
                'country': country_indicators,
                'metadata': {
                    'indicator_count': len(indicator_kpis),
                    'total_records': len(global_indicators) + len(country_indicators),
                    'countries_count': len(country_indicators['country'].unique()) if not country_indicators.empty else 0
                }
            }
            
        except Exception as e:
            return {'error': f'Indicator data extraction failed: {str(e)}'}
    
    def get_data_summary(self, unified_data: Dict) -> Dict:
        """Generate summary of extracted data."""
        summary = {
            'has_market_data': False,
            'has_indicator_data': False,
            'global_market_records': 0,
            'country_market_records': 0,
            'global_indicator_records': 0,
            'country_indicator_records': 0,
            'countries_with_data': [],
            'extraction_timestamp': 'unknown'
        }
        
        # Market data summary
        market_data = unified_data.get('market_value', {})
        if market_data and 'global' in market_data:
            summary['has_market_data'] = True
            summary['global_market_records'] = len(market_data['global'])
            summary['country_market_records'] = len(market_data.get('country', pd.DataFrame()))
            
            if not market_data.get('country', pd.DataFrame()).empty:
                summary['countries_with_data'] = list(market_data['country']['country'].unique())
        
        # Indicator data summary
        indicator_data = unified_data.get('indicators', {})
        if indicator_data and 'global' in indicator_data:
            summary['has_indicator_data'] = not indicator_data['global'].empty
            summary['global_indicator_records'] = len(indicator_data['global'])
            summary['country_indicator_records'] = len(indicator_data.get('country', pd.DataFrame()))
        
        # Extraction info
        extraction_params = unified_data.get('extraction_params', {})
        if extraction_params:
            summary['extraction_timestamp'] = extraction_params.get('extraction_timestamp', 'unknown')
        
        return summary
    
    def get_available_countries(self, unified_data: Dict) -> List[str]:
        """Get list of available countries from unified data."""
        countries = set()
        
        # From market data
        market_data = unified_data.get('market_value', {})
        if 'country' in market_data and not market_data['country'].empty:
            countries.update(market_data['country']['country'].unique())
        
        # From indicator data
        indicator_data = unified_data.get('indicators', {})
        if 'country' in indicator_data and not indicator_data['country'].empty:
            countries.update(indicator_data['country']['country'].unique())
        
        return sorted(list(countries))
    
    def has_database_connection(self) -> bool:
        """Check if database connection is available."""
        try:
            engine = get_engine()
            return engine is not None
        except Exception:
            return False
