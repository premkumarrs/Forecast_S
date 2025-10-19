"""
Data validation module for extraction parameters.
"""

from typing import Dict, List


class DataValidator:
    """Validates data extraction parameters."""
    
    def validate_params(self, market_kpi: str, indicator_kpis: List[str], 
                       hist_cutoff: int, forecast_until: int) -> Dict:
        """Validate extraction parameters."""
        errors = []
        warnings = []
        
        if not market_kpi.strip():
            errors.append("Market KPI is required")
        
        if hist_cutoff >= forecast_until:
            errors.append("Historical cutoff must be before forecast end year")
        
        if forecast_until - hist_cutoff > 20:
            warnings.append("Long forecast horizon (>20 years) may reduce accuracy")
        
        if not indicator_kpis:
            warnings.append("No indicator KPIs selected - only baseline forecasts will be available")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }