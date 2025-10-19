"""
Helper functions for Data Extraction page.
"""

import hashlib
from typing import List, Dict, Tuple


def get_cache_key(market_kpi: str, indicator_kpis: List[str], 
                 hist_cutoff: int, forecast_until: int) -> str:
    """
    Generate cache key for data extraction.
    
    Args:
        market_kpi: Market KPI
        indicator_kpis: List of indicator KPIs
        hist_cutoff: Historical cutoff year
        forecast_until: Forecast end year
        
    Returns:
        Cache key string
    """
    key_parts = [
        market_kpi,
        ','.join(sorted(indicator_kpis)),
        str(hist_cutoff),
        str(forecast_until)
    ]
    key_string = '|'.join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


def execute_unified_extraction(market_kpi: str, indicator_kpis: List[str], 
                              year_params: Dict) -> Tuple[Dict, bool]:
    """
    Execute unified data extraction.
    
    Args:
        market_kpi: Market KPI
        indicator_kpis: List of indicator KPIs
        year_params: Year parameters dictionary
        
    Returns:
        Tuple of (unified_data, success)
    """
    from src.services.data import DataExtractionService
    
    service = DataExtractionService()
    
    # Extract data
    unified_data = service.extract_unified_data(
        market_kpi=market_kpi,
        indicator_kpis=indicator_kpis,
        hist_cutoff=year_params['hist_cutoff'],
        forecast_until=year_params['forecast_until']
    )
    
    # Check for errors
    if 'error' in unified_data:
        return unified_data, False
    
    return unified_data, True