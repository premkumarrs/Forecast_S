"""Mock data generation for demo purposes."""

import pandas as pd
import numpy as np


def create_mock_data(query, params):
    """Create mock data for demo purposes when no DB is available"""
    # Determine query type from parameters
    if 'kpiKey_global_value' in params:
        # Global value data
        years = list(range(2016, params['hist_cutoff']))
        values = np.random.uniform(1000, 5000, len(years))
        return pd.DataFrame({
            'year': years,
            'value': values,
            'kpiKey': [params['kpiKey_global_value']] * len(years)
        })
    elif 'kpiKey_global_indicator' in params:
        # Global indicator data
        years = list(range(2016, params['forecast_until'] + 1))
        values = np.random.uniform(50, 150, len(years))
        return pd.DataFrame({
            'year': years,
            'value': values,
            'kpiKey': [params['kpiKey_global_indicator']] * len(years)
        })
    elif 'kpiKey_country_value' in params:
        # Country value data
        countries = ['United States', 'Germany', 'Japan', 'United Kingdom', 'France', 'China', 'India', 'Brazil']
        iso3_codes = ['USA', 'DEU', 'JPN', 'GBR', 'FRA', 'CHN', 'IND', 'BRA']
        years = list(range(2016, params['hist_cutoff']))
        
        data = []
        for i, country in enumerate(countries):
            for year in years:
                data.append({
                    'country': country,
                    'iso3': iso3_codes[i],
                    'year': year,
                    'value': np.random.uniform(100, 1000)
                })
        return pd.DataFrame(data)
    elif 'kpiKey_country_indicator' in params:
        # Country indicator data
        countries = ['United States', 'Germany', 'Japan', 'United Kingdom', 'France', 'China', 'India', 'Brazil']
        iso3_codes = ['USA', 'DEU', 'JPN', 'GBR', 'FRA', 'CHN', 'IND', 'BRA']
        years = list(range(2016, params['forecast_until'] + 1))
        
        data = []
        for i, country in enumerate(countries):
            for year in years:
                data.append({
                    'country': country,
                    'iso3': iso3_codes[i],
                    'year': year,
                    'value': np.random.uniform(50, 200)
                })
        return pd.DataFrame(data)
    
    return pd.DataFrame()