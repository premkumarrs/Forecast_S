"""Database utilities and helpers."""

import hashlib
from typing import Dict, Any


def get_cache_key(query_type: str, kpi_key: str, year_param: int) -> str:
    """Generate cache key for SQL queries"""
    return hashlib.md5(f"{query_type}_{kpi_key}_{year_param}".encode()).hexdigest()


def build_query_params(kpi_keys: list) -> Dict[str, Any]:
    """Build parameters dictionary for multiple KPI queries."""
    params = {}
    for i, kpi_key in enumerate(kpi_keys):
        params[f'kpi_{i}'] = kpi_key
    return params