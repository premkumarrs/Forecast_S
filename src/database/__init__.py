"""
Database module for SQL connections and data retrieval.
Simplified to use functions directly instead of classes.
"""

from .connection import query, get_engine, test_connection, DatabaseConnection
from .mock_data import create_mock_data
from .queries import build_kpi_query, build_multi_indicator_query
from .utils import get_cache_key, build_query_params

# Backward compatibility instance (deprecated - use query() directly)
db = DatabaseConnection()

__all__ = [
    'query', 'get_engine', 'test_connection',
    'db', 'DatabaseConnection',  # Backward compatibility
    'create_mock_data',
    'build_kpi_query', 'build_multi_indicator_query', 
    'get_cache_key', 'build_query_params'
]