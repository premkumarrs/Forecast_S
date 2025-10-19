"""
Mock data generator for demo purposes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime


class MockDataGenerator:
    """Generates mock data for testing and demos."""
    
    def create_mock_data(self, market_kpi: str, indicator_kpis: List[str], 
                        hist_cutoff: int, forecast_until: int) -> Dict:
        """Create mock data for demo purposes."""
        countries = ['United States', 'Germany', 'Japan', 'United Kingdom', 
                    'France', 'China', 'India', 'Brazil']
        iso3_codes = ['USA', 'DEU', 'JPN', 'GBR', 'FRA', 'CHN', 'IND', 'BRA']
        
        unified_data = {}
        
        # Mock Market Data (Historical only)
        years = list(range(2015, hist_cutoff + 1))
        global_market_data = []
        country_market_data = []
        
        # Global market data
        base_value = 50000
        for year in years:
            value = base_value * (1.05 ** (year - 2015)) + np.random.normal(0, base_value * 0.1)
            global_market_data.append({'year': year, 'value': max(0, value)})
        
        # Country market data
        for i, (country, iso3) in enumerate(zip(countries, iso3_codes)):
            country_base = base_value * (0.1 + i * 0.05)
            for year in years:
                value = country_base * (1.04 ** (year - 2015)) + np.random.normal(0, country_base * 0.15)
                country_market_data.append({
                    'country': country,
                    'iso3': iso3,
                    'year': year,
                    'value': max(0, value)
                })
        
        unified_data['market_value'] = {
            'global': pd.DataFrame(global_market_data),
            'country': pd.DataFrame(country_market_data),
            'metadata': {
                'kpi_key': market_kpi,
                'kpi_name': f'Mock {market_kpi}',
                'total_records': len(global_market_data) + len(country_market_data),
                'countries_count': len(countries)
            }
        }
        
        # Mock Indicator Data (includes forecast years)
        if indicator_kpis:
            all_years = list(range(2015, forecast_until + 1))
            global_indicator_data = []
            country_indicator_data = []
            
            for indicator in indicator_kpis:
                # Global indicators
                for year in all_years:
                    value = 100 + np.random.normal(0, 10) + (year - 2015) * 2
                    global_indicator_data.append({
                        'year': year,
                        'indicator_key': indicator,
                        'indicator_name': f'Mock {indicator}',
                        'value': value
                    })
                
                # Country indicators
                for country, iso3 in zip(countries, iso3_codes):
                    country_offset = np.random.normal(0, 5)
                    for year in all_years:
                        value = 100 + country_offset + np.random.normal(0, 8) + (year - 2015) * 1.5
                        country_indicator_data.append({
                            'country': country,
                            'iso3': iso3,
                            'year': year,
                            'indicator_key': indicator,
                            'indicator_name': f'Mock {indicator}',
                            'value': value
                        })
            
            unified_data['indicators'] = {
                'global': pd.DataFrame(global_indicator_data),
                'country': pd.DataFrame(country_indicator_data),
                'metadata': {
                    'indicator_count': len(indicator_kpis),
                    'total_records': len(global_indicator_data) + len(country_indicator_data),
                    'countries_count': len(countries)
                }
            }
        else:
            unified_data['indicators'] = {'global': pd.DataFrame(), 'country': pd.DataFrame()}
        
        # Extraction params
        unified_data['extraction_params'] = {
            'market_kpi': market_kpi,
            'indicator_kpis': indicator_kpis,
            'hist_cutoff': hist_cutoff,
            'forecast_until': forecast_until,
            'extraction_timestamp': datetime.now()
        }
        
        return unified_data